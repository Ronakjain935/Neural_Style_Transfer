import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth',
                        help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')
    
    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action=argparse.BooleanOptionalAction, default=True,
                        help='Crop image (pass --no-crop to disable)')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5.0,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
    
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    if len(content_dataset) == 0:
        raise ValueError(f"No valid images found in content_dir: '{args.content_dir}'")
    if len(style_dataset) == 0:
        raise ValueError(f"No valid images found in style_dir: '{args.style_dir}'")

    drop_last = len(content_dataset) >= args.batch_size and len(style_dataset) >= args.batch_size
    use_pin_memory = torch.cuda.is_available()

    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=use_pin_memory,
                                    drop_last=drop_last)
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=use_pin_memory,
                                  drop_last=drop_last)
    
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    # Freeze encoder parameters: evaluation mode and no gradients computed
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        if not args.decoder_path or not args.optimizer_path:
            raise ValueError("--resume requires both --decoder_path and --optimizer_path to be specified")
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))

    print('Training...')

    mse_loss = torch.nn.MSELoss()

    total_steps = min(len(content_dataloader), len(style_dataloader))
    if total_steps == 0:
        print("Warning: Dataloaders have 0 batches. Ensure batch_size <= dataset size.")
        return

    for epoch in range(args.epochs):
        decoder.train()
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=total_steps)

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        num_batches = 0

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            with torch.no_grad():
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)
                t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)
            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0.0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss: {loss.item():.4f}, Content Loss: {loss_c.item():.4f}, Style Loss: {loss_s.item():.4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            num_batches += 1
        
        scheduler.step()

        if num_batches > 0:
            running_loss /= num_batches
            running_closs /= num_batches
            running_sloss /= num_batches

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(f'Epoch {epoch+1}/{args.epochs}: Loss: {running_loss:.4f}, Content Loss: {running_closs:.4f}, Style Loss: {running_sloss:.4f}')

        is_last_epoch = (epoch + 1) == args.epochs
        if (epoch + 1) % args.save_interval == 0 or is_last_epoch:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                g_clamped = torch.clamp(g.detach(), 0.0, 1.0)
                output = torch.cat([content_batch, style_batch, g_clamped], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)




if __name__ == '__main__':
    main()
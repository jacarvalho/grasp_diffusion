import psutil
import os
import time
import datetime
import numpy as np
import torch
import gc
import subprocess

from collections import defaultdict
from se3dif.utils import makedirs, dict_to_device
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


process = psutil.Process(os.getpid())
memory_info = psutil.virtual_memory()

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, rank=0, max_steps=None, device='cpu'):
    
    print(f"Model device: {next(model.parameters()).device}")
    
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    # Build saving directories
    makedirs(model_dir)

    if rank == 0:
        summaries_dir = os.path.join(model_dir, 'summaries')
        makedirs(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        makedirs(checkpoints_dir)

        exp_name = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        writer = SummaryWriter(summaries_dir + '/' + exp_name)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []

        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))
            
            for step, (model_input, gt) in enumerate(train_dataloader):

                model_input = dict_to_device(model_input, device)
                gt = dict_to_device(gt, device)
                
                start_time = time.time()
                
                # Compute loss
                losses, iter_info = loss_fn(model, model_input, gt)
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss 
                train_losses.append(train_loss.item())
                
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # Save checkpoint
                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    if summary_fn is not None:
                        summary_fn(model, model_input, gt, iter_info, writer, total_steps)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                # Gradient clipping

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad if isinstance(clip_grad, float) else 1.0)

                for optim in optimizers:
                    optim.step()

                pbar.update(1)

                # Print iteration time and validation check
                test = total_steps % steps_til_summary
                if not total_steps % steps_til_summary and rank == 0:
                    print(f"Epoch {epoch}, Total loss {train_loss:.6f}, iteration time {time.time() - start_time:.6f}", end="")
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)

                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = dict_to_device(model_input, device)
                                gt = dict_to_device(gt, device)
                              
                                val_loss, val_iter_info = loss_fn(model, model_input, gt, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            if summary_fn is not None:
                                summary_fn(model, model_input, gt, val_iter_info, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        # cleanup to free up memory
                        gc.collect()
                        torch.cuda.empty_cache()

                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break
            
                if total_steps % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            print("")
            # print(f"RAM used: {process.memory_info().rss / 1024 ** 3:.4f} GB / Total: {memory_info.total / 1024 ** 3:.4f} GB")
            print(f"RAM used: {memory_info.used / 1024 ** 3:.4f} GB / Total: {memory_info.total / 1024 ** 3:.4f} GB")
            if torch.cuda.is_available():
                # print(f"GPU memory usage at end of epoch {epoch}: {torch.cuda.memory_allocated() / 1024 ** 3:.4f} GB")
                # print('\n'.join([f"GPU {i}: Used {int(used)/1024:.4f} GB / Total {int(total)/1024:.4f} GB" for i, (used, total) in enumerate([line.split(', ') for line in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], encoding='utf-8').strip().split('\n')])]))
                print('\n'.join([f"GPU used: {int(used)/1024:.4f} GB / Total {int(total)/1024:.4f} GB" for used, total in [line.split(', ') for line in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], encoding='utf-8').strip().split('\n')]]))

                
            if max_steps is not None and total_steps == max_steps:
                break

            if len(gc.garbage)>0:
                print(f"Uncollected objects: {len(gc.garbage)}")

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))

        return model, optimizers

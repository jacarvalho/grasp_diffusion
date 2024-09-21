import psutil
import os
import time
import datetime
import numpy as np
import torch
import gc

from collections import defaultdict
from se3dif.utils import makedirs, dict_to_device
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

process = psutil.Process(os.getpid())

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
        memory_start = process.memory_info().rss / 1024 ** 3
        GPU_memory_start = torch.cuda.memory_allocated() / 1024 ** 3
        # print(f"Memory usage before training loop: {process.memory_info().rss / 1024 ** 3:.4f} GB")
        
        print(f"RAM memory usage before training loop: {memory_start:.4f} GB")
        # print(f"GPU memory usage before training loop: {GPU_memory_start:.4f} GB")
        for epoch in range(epochs):
            memory_usage = process.memory_info().rss / 1024 ** 3
            GPU_memory_usage = torch.cuda.memory_allocated() / 1024 ** 3
            # print(f"Memory usage before epoch: {process.memory_info().rss / 1024 ** 3:.4f} GB")
            print('_'*50)
            print(f"RAM memory usage before epoch: {memory_usage:.4f} GB")
            # print(f"GPU memory usage before epoch: {GPU_memory_usage:.4f} GB")
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))
            
            for step, (model_input, gt) in enumerate(train_dataloader):
                # print(f"Memory usage before loading batch: {process.memory_info().rss / 1024 ** 2} MB")
                model_input = dict_to_device(model_input, device)
                gt = dict_to_device(gt, device)
                # print(f"Memory usage after loading batch: {process.memory_info().rss / 1024 ** 2} MB")
                
                continue
                start_time = time.time()
                # print('step', step)
                # print(f"Memory usage before loss computation: {process.memory_info().rss / 1024 ** 2} MB")
                # Compute loss
                losses, iter_info = loss_fn(model, model_input, gt)
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss 
                train_losses.append(train_loss.item())
                # print(f"Memory usage after loss computation: {process.memory_info().rss / 1024 ** 2} MB")
                
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # Save checkpoint
                # print(f"Memory usage before checkpoint: {process.memory_info().rss / 1024 ** 2} MB")
                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    if summary_fn is not None:
                        # print(f"Memory usage before summary_fn: {process.memory_info().rss / 1024 ** 2} MB")
                        summary_fn(model, model_input, gt, iter_info, writer, total_steps)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # print(f"Memory usage after summary_fn: {process.memory_info().rss / 1024 ** 2} MB")
                # print(f"Memory usage after checkpoint: {process.memory_info().rss / 1024 ** 2} MB")
                for optim in optimizers:
                    optim.zero_grad()
                # print(f"Memory usage before backward: {process.memory_info().rss / 1024 ** 2} MB")
                train_loss.backward()
                # print(f"Memory usage after backward: {process.memory_info().rss / 1024 ** 2} MB")
                # Gradient clipping
                # print(f"Memory usage before gradient clipping: {process.memory_info().rss / 1024 ** 2} MB")
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad if isinstance(clip_grad, float) else 1.0)
                # print(f"Memory usage after gradient clipping: {process.memory_info().rss / 1024 ** 2} MB")
                # print(f"Memory usage before optimizer step: {process.memory_info().rss / 1024 ** 2} MB")
                for optim in optimizers:
                    optim.step()
                # print(f"Memory usage after optimizer step: {process.memory_info().rss / 1024 ** 2} MB")
                pbar.update(1)

                # Print iteration time and validation check
                test = total_steps % steps_til_summary
                if not total_steps % steps_til_summary and rank == 0:
                    print(f"Epoch {epoch}, Total loss {train_loss:.6f}, iteration time {time.time() - start_time:.6f}", end="\n")
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            # print(f"Memory usage before validation: {process.memory_info().rss / 1024 ** 2} MB")
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                # print('Validation step', val_i)
                                # print(f"Memory usage before validation step: {process.memory_info().rss / 1024 ** 2} MB")
                                model_input = dict_to_device(model_input, device)
                                gt = dict_to_device(gt, device)
                                # print(f"Memory usage after loading validation batch to device: {process.memory_info().rss / 1024 ** 2} MB")

                                # print(f"Memory usage before loss_fn: {process.memory_info().rss / 1024 ** 2} MB")
                                val_loss, val_iter_info = loss_fn(model, model_input, gt, val=True)
                                # print(f"Memory usage after loss function: {process.memory_info().rss / 1024 ** 2} MB")

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                            # print(f"Memory usage after validation: {process.memory_info().rss / 1024 ** 2} MB")
                        
                        # print(f"Memory usage before validation loss computation: {process.memory_info().rss / 1024 ** 2} MB")
                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            if summary_fn is not None:
                                # print(f"Memory usage before summary_fn: {process.memory_info().rss / 1024 ** 2} MB")
                                summary_fn(model, model_input, gt, val_iter_info, writer, total_steps, 'val_')
                                # print(f"Memory usage after summary_fn: {process.memory_info().rss / 1024 ** 2} MB")
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)
                        # print(f"Memory usage after validation loss computation: {process.memory_info().rss / 1024 ** 2} MB")

                        # cleanup to free up memory
                        gc.collect()
                        torch.cuda.empty_cache()
                        # print(f"Memory usage after validation cleanup: {process.memory_info().rss / 1024 ** 2} MB")

                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break
            
            if total_steps % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            print(f"RAM memory usage after epoch: {process.memory_info().rss / 1024 ** 3:.4f} GB")
            print(f"RAM memory diff after epoch: {(process.memory_info().rss / 1024 ** 3 - memory_usage):.4f} GB")
            print(f"RAM total memory diff: {(process.memory_info().rss / 1024 ** 3 - memory_start):.4f} GB")
            
            # print(f"GPU memory usage after epoch (allocated): {torch.cuda.memory_allocated() / 1024 ** 3:.4f} GB")
            # print(f"GPU memory usage after epoch (reserved): {torch.cuda.memory_reserved() / 1024 ** 3:.4f} GB")
            # print(f"GPU memory diff after epoch (allocated): {(torch.cuda.memory_allocated() / 1024 ** 3 - GPU_memory_usage):.4f} GB")
            # print(f"GPU memory diff after epoch (reserved): {(torch.cuda.memory_reserved() / 1024 ** 3 - GPU_memory_usage):.4f} GB")

            
            if max_steps is not None and total_steps == max_steps:
                break

            if len(gc.garbage)>0:
                print(f"Uncollected objects: {len(gc.garbage)}")

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))

        return model, optimizers

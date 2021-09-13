# NetQuantz
Neural Network Quantization 

## Pytorch 1.9 compatibility changes
### Resolved
* changed `correct_k = correct[:k].view(-1).float().sum(0)` to `correct_k = correct[:k].reshape(-1).float().sum(0)` in accuracy function in utils.py.
* Solved the following by using `with_no_grad` instead of using deprecated `volatile` argument: `my_main.py:319: UserWarning: volatile was removed and now has no effect. Use "with torch.no_grad():" instead.` 
* Got rid of these warnings by using boolean operators rather than literals: `my_main.py:106: SyntaxWarning: "is" with a literal. Did you mean "=="? if args.save is '':` and `my_main.py:127: SyntaxWarning: "is not" with a literal. Did you mean "!="? if args.model_config is not '':`
* 2. Got rid of warning by using transforms.Resize instead of transforms.Scale: `UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead. warnings.warn("The use of the transforms.Scale transform is deprecated, "`

### Random warnings due to imports
* `[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)`: This seems to be safe to ignore, and will apparently be muted in a future Pytorch release: [Link](https://discuss.pytorch.org/t/warning-leaking-caffe2-thread-pool-after-fork-function-pthreadpool/127559)
* `UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.`: This seems to be an issue with torchvision.datasets. Should be safe to ignore: [Link](https://githubmemory.com/repo/pytorch/vision/issues/4183)
* `UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable.`: Not sure about this one.

### TODO:
* Test that results still make sense. Compare against 0.4.1 version.
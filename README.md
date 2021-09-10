# NetQuantz
Neural Network Quantization 

### Pytorch 1.9 compatibility changes
Resolved
* changed `correct_k = correct[:k].view(-1).float().sum(0)` to `correct_k = correct[:k].reshape(-1).float().sum(0)` in accuracy function in utils.py.
* Solved the following by using `with_no_grad` instead of using deprecated `volatile` argument: `my_main.py:319: UserWarning: volatile was removed and now has no effect. Use "with torch.no_grad():" instead.` 
* Got rid of these warnings by using boolean operators rather than literals: `my_main.py:106: SyntaxWarning: "is" with a literal. Did you mean "=="? if args.save is '':` and `my_main.py:127: SyntaxWarning: "is not" with a literal. Did you mean "!="? if args.model_config is not '':`

TODO:
1. `[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)`
2. `UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.`
3. `UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable.`

Number (1) seems to be safe to ignore, and will apparently be muted in a future Pytorch release: [Link](https://discuss.pytorch.org/t/warning-leaking-caffe2-thread-pool-after-fork-function-pthreadpool/127559)

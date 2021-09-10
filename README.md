# NetQuantz
Neural Network Quantization 

### Pytorch 1.9 compatibility changes
Resolved
* changed `correct_k = correct[:k].view(-1).float().sum(0)` to `correct_k = correct[:k].reshape(-1).float().sum(0)` in accuracy function in utils.py.
TODO:
1. `my_main.py:319: UserWarning: volatile was removed and now has no effect. Use "with torch.no_grad():" instead.`
2. `my_main.py:106: SyntaxWarning: "is" with a literal. Did you mean "=="? if args.save is '':`
3. `my_main.py:127: SyntaxWarning: "is not" with a literal. Did you mean "!="? if args.model_config is not '':`
4. `[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)`

Number (4) seems to be safe to ignore, and will apparently be muted in a future Pytorch release: [Link](https://discuss.pytorch.org/t/warning-leaking-caffe2-thread-pool-after-fork-function-pthreadpool/127559)

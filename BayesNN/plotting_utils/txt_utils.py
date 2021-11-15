

def parse_md_tanh_log_file(file_path):
    f = open(file_path)
    lines = f.readlines()
    losses, accs  = [], []
    for line in lines:
        if 'Loss: ' in line: 
            sp = line.split(',')
            loss = float(sp[2].split(' ')[2])
            acc = float(sp[3].split(' ')[2])
            losses.append(loss)
            accs.append(acc)

    return losses, accs

def parse_bayesbinn_log_file(file_path):
    f = open(file_path)
    lines = f.readlines()
    losses, accs  = [], []

    for line in lines:
        if 'Train Loss: ' in line: 
            sp = line.split(',')
            loss = float(sp[1].split(' ')[3])
            # acc = float(sp[3].split(' ')[2])
            losses.append(loss)
            # accs.append(acc)
        elif 'Test Accuracy: ' in line:
            sp = line.split(',')
            acc = float(sp[1].split(' ')[13])
            accs.append(acc)

    return losses, accs

    

if __name__ == '__main__':
    #file_path = '/home/intellhave/Work/RunResults/SelectedRuns/CIFAR10/VGG16/BC/log.txt'
    #losses, accs=parse_log_file(file_path)

    file_path = '/home/intellhave/Work/RunResults/SelectedRuns/CIFAR10/RESNET18/FENBP/log.txt'
    losses, accs=parse_bayesbinn_log_file(file_path)
    print(losses)
    print(accs)

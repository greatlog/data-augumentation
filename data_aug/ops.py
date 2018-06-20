import os

def generate_list(flags):
    label_kind = os.listdir(flags.data_dir)
    label_kind.sort()
    file_list = []
    label_list = []
    for root, dirs, files in os.walk(flags.data_dir):
        for file in files:
            name = os.path.join(root, file)
            label_str = name.split('/')[8]
            for i in range(flags.class_num):
                if label_str == label_kind[i]:
                    label_list.append(i)
            file_list.append(os.path.join(root, file))

    train_list = {'data':file_list[:flags.data_split],'label':label_list[:flags.data_split]}
    test_list = {'data':file_list[flags.data_split:-1],'label':label_list[flags.data_split:-1]}

    return label_kind, train_list, test_list

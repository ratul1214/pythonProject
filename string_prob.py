sentence = 'i am saud. hello world'
num = 200
if len(sentence) < num:
    print(sentence)
else:
    cutted = sentence[0:num+1]
    if cutted[len(cutted)-1] != ' ':

        #cutted.rsplit(' ', 1)[-1]
        new = ' '.join(cutted.split(' ')[:-1])
        print(new)
    else:
        cutted = cutted[0:num]
        print(cutted)

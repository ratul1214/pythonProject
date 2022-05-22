mainString ='abcdabl'
maxString = ''
maxStringlen = 0;
tempString = ''
for char in mainString:
    if tempString.find(char) == -1:
        tempString = tempString+char
    else:
        if len(tempString) > maxStringlen:
            maxStringlen = len(tempString)
            maxString = tempString
        tempString = ''
        tempString = tempString+char

if len(tempString) > maxStringlen:
    maxStringlen = len(tempString)
    maxString = tempString
print(maxStringlen)
print(maxString)
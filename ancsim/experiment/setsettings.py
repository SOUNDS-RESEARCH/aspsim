import sys

def changeParameter(name, val):
    print("name: ", name)
    print("val: ", val)
    with open("settings.py","r") as file:
        lineIdx = None
        text = file.readlines()
        for i, line in enumerate(text):
            words = line.split()
            if len(words) > 0:
                if words[0] == name:
                    lineIdx = i
        assert (lineIdx != None)

    text[lineIdx] = name + " = " + str(val) + "\n"
    with open("settings.py","w") as file:
        file.writelines(text)

     

def main():
    print(sys.argv)
    params = sys.argv[1:]
    #print("numparams ", numParams)
    print("Sys.argv length: ",len(sys.argv))
    
    name = params[0]
    val = None
    for p in range(1,len(params)):
        newVal = getNum(params[p])
        if isinstance(newVal, (int, float)):
            if val is None:
                val = newVal
            elif isinstance(val,(int,float)):
                val = [val, newVal]
            else:
                val.append(newVal)
        elif isinstance(newVal, (str)):
            if isinstance(val, list):
                val = tuple(val)
            changeParameter(name, val)
            name = newVal
            val = None

    changeParameter(name, val)
        

def getNum(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            val = c(val)
            return val
        except ValueError:
            pass

if __name__ == "__main__":
    main()
    


def func():
    print("hi")


# while testing, try to keep everything inside of a block like this
# these won't get called when the module is imported but it will be called
# if you try to run the file
if __name__ == "__main__":
    func()
    a = 3
    b = 5
    # etc.
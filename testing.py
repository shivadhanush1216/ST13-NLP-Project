from test import TextToNum
ob=TextToNum("coding is good, but hard to learn!!")
ob.cleaner()
ob.token()
ob.removeStop()
dt=ob.stemme()
print(dt)
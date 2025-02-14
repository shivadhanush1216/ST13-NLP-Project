from test import TextToNum  

ob = TextToNum("coding is good, but hard to learn!!")  # Pass text while creating the object
cleaned_text = ob.cleaner()  # Call the cleaner method
print(cleaned_text)  # Output the result
ob.cleaner()
ob.token()
ob.removeStop()
dt=ob.stemme()
print(dt)
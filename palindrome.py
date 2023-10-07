s=input("Enter string")
s1=""
for i in s:
    s1=i+s1
if s1==s:
    print("PALINDROME")
else:
    print("NOT A PALINDROME")
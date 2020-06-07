import os
import matplotlib
import matplotlib.pyplot as plt
print (os.getcwd()+'\\Data\\output\\xte_shuffled.npy')
print(os.path.abspath(os.getcwd()))
print(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__))+'\\Data\\output\\xte_shuffled.npy')
print(os.path.dirname(os.path.realpath(__file__)))
##os.path.dirname(os.path.realpath(__file__))+"\\Data\\Input\\"
if not os.path.isfile(os.path.abspath(os.getcwd())+'\\Data\\input\\xte_shuffled.npy'):
    print(os.path.dirname(os.path.realpath(__file__)))
    print("Please clean the data first")
plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
plt.xlabel('Months')
#plt.show()
plt.savefig(os.path.abspath(os.getcwd())+'\\Data\\Output\\'+str(1)+'books_read.png')
import scipy.io

# Load the .mat file
file = scipy.io.loadmat('glass.mat')

# Assuming the data is stored under keys 'X' and 'y'
X_data = file['X']
y_data = file['y']

# Displaying the first 5 entries of each
print("First 5 entries of X:")
print(X_data[:5])
print("\nFirst 5 entries of y:")
print(y_data[:5])




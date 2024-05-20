


import numpy as np



# Example complex arrays
a = np.array([1+2j, 2+3j, 3+4j, 4+5j])
b = np.array([2+10j, 3+2j])
c = np.array([2+3j, 3+4j])

# Compute the cross-correlation
correlation  = np.correlate(a, np.conj(b), mode='valid')
correlation2 = np.correlate(a, np.conj(c), mode='valid')

# Print the result
print("Cross-correlation of a and b:", np.abs(correlation))
print("Cross-correlation of a and c:", np.abs(correlation2))
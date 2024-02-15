import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define universe variables
age = np.arange(0, 100, 1)
blood_pressure = np.arange(90, 200, 1)
cholesterol = np.arange(120, 580, 1)
heart_rate = np.arange(40, 200, 1)

# Define parameters for low and high regions for each variable
age_low = 0
age_high = 100

blood_pressure_low = 90
blood_pressure_high = 200

cholesterol_low = 120
cholesterol_high = 580

heart_rate_low = 40
heart_rate_high = 200

# Generate fuzzy membership functions using Gaussian membership functions
age_membership_low = fuzz.gaussmf(age, age_low, 30.37425780967358)
age_membership_high = fuzz.gaussmf(age, age_high, 98.92323148607927)

blood_pressure_membership_low = fuzz.gaussmf(blood_pressure, blood_pressure_low, 31.30203993566636)
blood_pressure_membership_high = fuzz.gaussmf(blood_pressure, blood_pressure_high, 88.35776635756176)

cholesterol_membership_low = fuzz.gaussmf(cholesterol, cholesterol_low, 71.2218471050781)
cholesterol_membership_high = fuzz.gaussmf(cholesterol, cholesterol_high, 43.734075579319246)

heart_rate_membership_low = fuzz.gaussmf(heart_rate, heart_rate_low, 67.63996847971873)
heart_rate_membership_high = fuzz.gaussmf(heart_rate, heart_rate_high, 30.937387194856644)

# Visualize membership functions
plt.figure(figsize=(12, 10))

plt.subplot(421)
plt.plot(age, age_membership_low, 'b', linewidth=1.5, label='Low')
plt.plot(age, age_membership_high, 'r', linewidth=1.5, label='High')
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('Membership')
plt.legend()

plt.subplot(422)
plt.plot(blood_pressure, blood_pressure_membership_low, 'b', linewidth=1.5, label='Low')
plt.plot(blood_pressure, blood_pressure_membership_high, 'r', linewidth=1.5, label='High')
plt.title('Blood Pressure')
plt.xlabel('Blood Pressure')
plt.ylabel('Membership')
plt.legend()

plt.subplot(423)
plt.plot(cholesterol, cholesterol_membership_low, 'b', linewidth=1.5, label='Low')
plt.plot(cholesterol, cholesterol_membership_high, 'r', linewidth=1.5, label='High')
plt.title('Cholesterol')
plt.xlabel('Cholesterol')
plt.ylabel('Membership')
plt.legend()

plt.subplot(424)
plt.plot(heart_rate, heart_rate_membership_low, 'b', linewidth=1.5, label='Low')
plt.plot(heart_rate, heart_rate_membership_high, 'r', linewidth=1.5, label='High')
plt.title('Heart Rate')
plt.xlabel('Heart Rate')
plt.ylabel('Membership')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import math

def gaussian(x, mu, sigma):
    """
    Calculate the Gaussian function value for a given x, mean, and standard deviation.
    """
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def calculate_mse(actual_output, calculated_output):
    """
    Calculate the Mean Squared Error (MSE) between the actual and predicted output.
    """
    return 1 * (actual_output - calculated_output) ** 2\

def tanh(x):
    """
    Tan Sigmoid function used as activation function to get output between 0-1.
    """
    return math.tanh(1 / 400 * x)

def calculate_mse(actual_output, calculated_output):
    """
    Calculate the Mean Squared Error (MSE) between the actual and predicted output.
    """
    return 1 * (actual_output - calculated_output) ** 2\

def ANFIS(sd_values, a_values, b_values, c_values, d_values, e_values, age, blood_pressure, cholesterol, heart_rate):
    """
    ANFIS
    """

    # Specified Mean Values
    mean_values = [0, 100, 90, 200, 120, 580, 40, 200]

    gaussian_parameters = {
        'age': {
            'low': {'mu': mean_values[0], 'sigma': sd_values[0]},
            'high': {'mu': mean_values[1], 'sigma': sd_values[1]}
        },
        'blood_pressure': {
            'low': {'mu': mean_values[2], 'sigma': sd_values[2]},
            'high': {'mu': mean_values[3], 'sigma': sd_values[3]}
        },
        'cholesterol': {
            'low': {'mu': mean_values[4], 'sigma': sd_values[4]},
            'high': {'mu': mean_values[5], 'sigma': sd_values[5]}
        },
        'heart_rate': {
            'low': {'mu': mean_values[6], 'sigma': sd_values[6]},
            'high': {'mu': mean_values[7], 'sigma': sd_values[7]}
        }
    }

    # Calculate the membership values
    age_low = gaussian(age, **gaussian_parameters['age']['low'])
    age_high = gaussian(age, **gaussian_parameters['age']['high'])

    bp_low = gaussian(blood_pressure, **gaussian_parameters['blood_pressure']['low'])
    bp_high = gaussian(blood_pressure, **gaussian_parameters['blood_pressure']['high'])

    cholesterol_low = gaussian(cholesterol, **gaussian_parameters['cholesterol']['low'])
    cholesterol_high = gaussian(cholesterol, **gaussian_parameters['cholesterol']['high'])

    heart_rate_low = gaussian(heart_rate, **gaussian_parameters['heart_rate']['low'])
    heart_rate_high = gaussian(heart_rate, **gaussian_parameters['heart_rate']['high'])

    # Calculate weights
    weights = {
        'w1': age_low * cholesterol_low * bp_low * heart_rate_low,
        'w2': age_low * cholesterol_low * bp_low * heart_rate_high,
        'w3': age_low * cholesterol_low * bp_high * heart_rate_low,
        'w4': age_low * cholesterol_low * bp_high * heart_rate_high,
        'w5': age_low * cholesterol_high * bp_low * heart_rate_low,
        'w6': age_low * cholesterol_high * bp_low * heart_rate_high,
        'w7': age_low * cholesterol_high * bp_high * heart_rate_low,
        'w8': age_low * cholesterol_high * bp_high * heart_rate_high,
        'w9': age_high * cholesterol_low * bp_low * heart_rate_low,
        'w10': age_high * cholesterol_low * bp_low * heart_rate_high,
        'w11': age_high * cholesterol_low * bp_high * heart_rate_low,
        'w12': age_high * cholesterol_low * bp_high * heart_rate_high,
        'w13': age_high * cholesterol_high * bp_low * heart_rate_low,
        'w14': age_high * cholesterol_high * bp_low * heart_rate_high,
        'w15': age_high * cholesterol_high * bp_high * heart_rate_low,
        'w16': age_high * cholesterol_high * bp_high * heart_rate_high,
    }

    # Calculate the sum of all weights
    total_weight = sum(weights.values())

    # Return infinity if weight is 0
    if total_weight == 0:
        return float('inf')

    # Normalize the weights
    normalized_weights = {key: value / total_weight for key, value in weights.items()}

    # Calculate O1 to O16 using Sugeno Fuzzy
    O_values = [normalized_weights[f'w{i}'] * (
            a_values[i - 1] * age + b_values[i - 1] * blood_pressure + c_values[i - 1] * cholesterol + d_values[
        i - 1] * heart_rate + e_values[i - 1]) for i in range(1, 17)]

    # Sum up all the output values
    total_output = sum(O_values)

    # Calculate the tanh/tan sigmoid of total_output
    tanh_output = tanh(total_output)

    return tanh_output

# Optimized Parameters taken from ANFIS/GA Training

sd_values = [30.37425780967358, 98.92323148607927, 31.30203993566636, 88.35776635756176, 71.2218471050781,
                 43.734075579319246, 67.63996847971873, 30.937387194856644]
a_values = [0.9732031435551686, 0.002688074077581315, 0.7943203599039085, 0.026442885173838748, 0.2738865067022688,
                0.3233673717050167, 0.9236541443806265, 0.14600392757227287, 0.43551706252478106, 0.04351124072960244,
                0.4098028348551067, 0.06671524745501212, 0.9460516326944615, 0.39275483342725104, 0.043528052946073426,
                0.18453543291968943]
b_values = [0.4432199392040913, 0.037194388229018105, 0.2570273196928561, 0.09723134452831073, 0.16818769955303936,
                0.25895733891737793, 0.1028899705491454, 0.7859214440275468, 0.04305857294284565, 0.09831302374188688,
                0.7597817056067317, 0.4294200092367515, 0.4598972809917097, 0.8499582199532707, 0.8219892834888435,
                0.2801227974078886]
c_values = [0.2365433657016054, 0.2001727628887343, 0.961140048391374, 0.10536567826209697, 0.5158219868486323,
                0.33785117362711414, 0.6735582930337685, 0.21666303634202333, 0.05949757890955576, 0.14326558874800255,
                0.9394102696667682, 0.012812112251490704, 0.7454729540633036, 0.6273417417146013, 0.41563968337136015,
                0.8305842718545331]
d_values = [0.06136473359583394, 0.04385758888744107, 0.5255328522142582, 0.008550749064928032, 0.2074167659044439,
                0.4497875055211833, 0.3425908291834572, 0.19384831046407147, 0.7434463052710825, 0.07912695990173457,
                0.8512781317591239, 0.03912567423121471, 0.9369364752103462, 0.16299795840068443, 0.02071330803936222,
                0.3527960988800706]
e_values = [0.23601646711979352, 0.12293345617954032, 0.8580113475436795, 0.01656766283823785, 0.4637885329033492,
                0.7931768977953763, 0.15894282957550965, 0.9391566238537904, 0.21034956262735371, 0.027206347898655836,
                0.1913517578065863, 0.016095303165399644, 0.8594526302435292, 0.9266191954170502, 0.7592609496608976,
                0.08776750692808122]

# Define universe variables
age = np.arange(0, 100, 1)
blood_pressure = np.arange(90, 200, 1)
cholesterol = np.arange(120, 580, 1)
heart_rate = np.arange(40, 200, 1)

# Define the layout of the GUI
layout = [
    [sg.Text('Enter your age (0-100):'), sg.InputText(key='age')],
    [sg.Text('Enter your blood pressure (90-200):'), sg.InputText(key='bp')],
    [sg.Text('Enter your cholesterol level (120-580):'), sg.InputText(key='ch')],
    [sg.Text('Enter your max heart rate (40-200):'), sg.InputText(key='hr')],
    [sg.Button('Check Heart Attack Risk'), sg.Button('Exit')],
    [sg.Text('Result:'), sg.Text('', size=(10, 1), key='result')],
    [sg.Text('Notes:'), sg.Text('', size=(30, 1), key='notes')]
]

# Create the GUI window
window = sg.Window('Heart Attack Risk Detector - Powered by ANFIS-GA', layout)

def visualize_fuzzy(age_val, bp_val, ch_val, hr_val):
    # Generate fuzzy membership functions
    age_membership_low = fuzz.gaussmf(age, 0, 30.37425780967358)
    age_membership_high = fuzz.gaussmf(age, 100, 98.92323148607927)

    blood_pressure_membership_low = fuzz.gaussmf(blood_pressure, 90, 31.30203993566636)
    blood_pressure_membership_high = fuzz.gaussmf(blood_pressure, 200, 88.35776635756176)

    cholesterol_membership_low = fuzz.gaussmf(cholesterol, 120, 71.2218471050781)
    cholesterol_membership_high = fuzz.gaussmf(cholesterol, 580, 43.734075579319246)

    heart_rate_membership_low = fuzz.gaussmf(heart_rate, 40, 67.63996847971873)
    heart_rate_membership_high = fuzz.gaussmf(heart_rate, 200, 30.937387194856644)

    # Visualize membership functions with user input highlighted
    plt.figure(figsize=(12, 10))

    plots = [
        (age, age_membership_low, age_membership_high, age_val, 'Age'),
        (blood_pressure, blood_pressure_membership_low, blood_pressure_membership_high, bp_val, 'Blood Pressure'),
        (cholesterol, cholesterol_membership_low, cholesterol_membership_high, ch_val, 'Cholesterol'),
        (heart_rate, heart_rate_membership_low, heart_rate_membership_high, hr_val, 'Heart Rate')
    ]

    for i, (var, mem_low, mem_high, val, title) in enumerate(plots, start=1):
        plt.subplot(4, 1, i)
        plt.plot(var, mem_low, 'b', linewidth=1.5, label='Low')
        plt.plot(var, mem_high, 'r', linewidth=1.5, label='High')
        plt.axvline(x=val, color='g', linestyle='--')
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel('Membership')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Event loop to process events and get values from the GUI
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    if event == 'Check Heart Attack Risk':
        try:
            age_val = float(values['age'])
            bp_val = float(values['bp'])
            ch_val = float(values['ch'])
            hr_val = float(values['hr'])
            output = ANFIS(sd_values, a_values, b_values, c_values, d_values, e_values, age_val, bp_val, ch_val, hr_val)
            if output<0.3:
                notes = "Low risk of heart attack"
            elif output>0.3 and output<0.7:
                notes = "Moderate risk of heart attack"
            else:
                notes = "High risk of heart attack"

            visualize_fuzzy(age_val, bp_val, ch_val, hr_val)

            window['result'].update(output)
            window['notes'].update(notes)
        except ValueError:
            sg.popup_error('Please enter valid numbers.')

# Close the GUI window
window.close()

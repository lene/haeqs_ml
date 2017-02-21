
# pip install pandas scikit-learn scipy numpy
import pandas as pd
from sklearn.linear_model import Ridge as Classifier

# sudo apt install python3-tk
# pip install matplotlib
import matplotlib.pyplot as plt

house_data = pd.read_csv('house_data.csv')[:100]

house_data['sqm_living'] = house_data['sqft_living']*0.092903

prices = house_data['price']
features = house_data[['sqm_living']]
classifier = Classifier()
classifier.fit(features, prices)
print('100 sqm', classifier.predict(100))
print('100 sqm', classifier.predict(200))

print(house_data[:10])

# prices.plot(x='sqft_living', y='price')
# plt.show()
print(house_data.keys())

features = house_data[['sqm_living', 'bedrooms']]
classifier = Classifier()
classifier.fit(features, prices)
print('100 sqm, 1 bedroom', classifier.predict([[100, 1]]))
print('200 sqm, 2 bedrooms', classifier.predict([[200, 2]]))


features = house_data[['sqm_living', 'bedrooms', 'waterfront']]
classifier = Classifier()
classifier.fit(features, prices)
print('boring ass house', classifier.predict([[200, 2, 0]]))
print('house at water', classifier.predict([[200, 2, 1]]))



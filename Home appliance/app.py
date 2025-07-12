from flask import Flask, render_template, request
from utils import predict_prices

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    breakdown = []

    if request.method == 'POST':
        appliances = []
        for i in range(1, 5):  # 4 appliance slots
            type_ = request.form.get(f'type{i}')
            brand = request.form.get(f'brand{i}')
            capacity = request.form.get(f'capacity{i}')
            stars = request.form.get(f'star{i}')
            feature = request.form.get(f'feature{i}')

            if type_ and brand:
                appliances.append({
                    'appliance_type': type_,
                    'brand': brand,
                    'capacity_l': float(capacity) if capacity else None,
                    'star_rating': float(stars) if stars else None,
                    'features': feature
                })

        if appliances:
            prices = predict_prices(appliances)
            total = sum(prices)
            breakdown = list(zip(appliances, prices))
            prediction = f"Estimated Total Budget: ₹{round(total):,}"

    return render_template("index.html", prediction=prediction, breakdown=breakdown)
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)


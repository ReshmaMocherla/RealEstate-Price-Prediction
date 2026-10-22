# Bengaluru House Price Prediction

This is a machine learning project I worked on to predict house prices in Bengaluru based on a few property details, such as location, area in square feet, number of bedrooms (BHK), and bathrooms.

I connected a regression model to a Flask backend and built a simple web interface where users can enter these details and get an estimated price. The project helped me understand how a machine learning model can be used in a web application instead of running predictions only inside a notebook.

**GitHub Repository:** [RealEstate-Price-Prediction](https://github.com/ReshmaMocherla/RealEstate-Price-Prediction)

## Project Preview

![Bengaluru House Price Prediction](BHP/ResultPage.png)

## Technologies Used

- **Python** – model development and backend logic
- **Pandas and NumPy** – working with housing data and model inputs
- **scikit-learn** – building the regression model
- **Flask** – handling requests from the frontend
- **HTML, CSS and JavaScript** – building the user interface
- **jQuery and HTTP** – sending requests to the backend
- **JSON** – exchanging data between the frontend and backend
- **Jupyter Notebook** – exploring the dataset and developing the model
- **Postman** – testing API requests

## How the Application Works

The application has three main parts:

1. **Machine learning model:** The housing dataset is explored in a Jupyter notebook, and a regression model is trained to estimate property prices.
2. **Flask backend:** The backend provides endpoints to retrieve available locations and generate price predictions using the saved model.
3. **Web interface:** Users enter property details, submit the form, and see the estimated price returned by the backend.

### Application Flow

```text
User enters property details
          |
          v
    Web Interface
          |
          | HTTP Request
          v
     Flask Backend
          |
          v
   Prediction Utility
          |
          v
   Trained ML Model
          |
          v
   Estimated Price
          |
          v
    JSON Response
          |
          v
  Result shown to user
```

## API Endpoints

The Flask application has two main endpoints.

### 1. Get available locations

**Method:** `GET`

**Endpoint:** `/get_location_names`

This endpoint returns the available locations used by the application. The frontend uses the response to populate the location dropdown.

Example request:

```http
GET http://127.0.0.1:5000/get_location_names
```

Example response structure:

```json
{
  "locations": [
    "1st Phase JP Nagar",
    "Electronic City",
    "Rajaji Nagar"
  ]
}
```

These locations are examples to illustrate the response format. The actual values come from the project's saved metadata.

### 2. Predict a house price

**Method:** `POST`

**Endpoint:** `/predict_home_price`

This endpoint receives the property details from the frontend and passes them to the prediction utility.

The request uses form data with the following fields:

| Field | Description |
|---|---|
| `total_sqft` | Total area of the property |
| `bhk` | Number of bedrooms |
| `bath` | Number of bathrooms |
| `location` | Selected property location |

Example response structure:

```json
{
  "estimated_price": 50.0
}
```

The value above is only an example, not a verified prediction from the model. The application returns the estimated price calculated from the submitted inputs.

## Project Structure

```text
RealEstate-Price-Prediction/
├── README.md
└── BHP/
    ├── ResultPage.png
    ├── client/
    │   ├── app.html
    │   ├── app.css
    │   └── app.js
    ├── model/
    │   ├── RealEstate-PricePrediction.ipynb
    │   ├── Columns.json
    │   └── banglore_home_prices_model.pickle
    └── server/
        ├── server.py
        └── util.py
```

## Running the Project Locally

### Prerequisites

- Python and pip
- Git
- A web browser

### 1. Clone the repository

```bash
git clone https://github.com/ReshmaMocherla/RealEstate-Price-Prediction.git
cd RealEstate-Price-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install flask numpy scikit-learn
```

The notebook may require additional packages, such as Pandas and Matplotlib.

### 4. Check the model files

The backend needs `Columns.json` and `banglore_home_prices_model.pickle` to load the saved model and its feature information.

Currently, these files are in `BHP/model/`, while `util.py` expects them in an `artifacts` directory relative to its working directory. The paths need to be aligned before running the application—for example, by updating the paths in `util.py` or placing the files in the expected directory.

### 5. Start the Flask backend

After resolving the model file paths, open a terminal in `BHP/server` and run:

```bash
python server.py
```

The application is configured to run locally on port `5000`.

### 6. Open the frontend

Open `BHP/client/app.html` in a browser after starting the backend.

The frontend sends requests to:

```text
http://127.0.0.1:5000
```

If the requests fail when opening the HTML file directly, serve the frontend locally and check the browser console and Flask logs for any errors.

## What I Learned

While working on this project, I gained practical experience with:

- Preparing housing data for a machine learning model.
- Saving and loading a trained model for later use.
- Creating Flask endpoints for GET and POST requests.
- Passing form data from JavaScript to a Python backend.
- Returning prediction results in JSON format.
- Using Postman to inspect API requests and responses.

## Possible Improvements

Some areas I would like to improve are:

- Adding input validation and better error handling.
- Writing automated tests for the API and prediction utility.
- Documenting the model's evaluation metrics and prediction accuracy.
- Managing dependencies in a requirements file.
- Improving the setup process and deploying the application.

## Acknowledgements

The dataset source and any tutorials or learning resources used for this project should be credited here.

## Author

**Reshma Mocherla**

[GitHub Profile](https://github.com/ReshmaMocherla)

*This project is intended for learning and demonstration. The predicted prices are estimates and should not be treated as professional property valuations.*

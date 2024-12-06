# Phishing URL Detector

A Chrome Extension that highlights potential phishing emails or malicious URLs in the browser using Machine Learning.

---

## Project Plan

### 1. Define Objectives
- Build a Chrome Extension to highlight phishing URLs.
- Use a machine learning model to classify URLs and detect phishing attempts.

### 2. Technology Stack
#### Frontend (Chrome Extension)
- **Languages**: HTML, CSS, JavaScript
#### Backend (ML API)
- **Framework**: Python with Flask or FastAPI
- **Libraries**: Scikit-learn, TensorFlow, PyTorch, `tldextract`, `re`
#### Data
- Use datasets like [PhishTank](https://phishtank.org/) or [OpenPhish](https://openphish.com/).

### 3. Core Functionalities
- **Browser Integration**: Monitor page content and extract URLs.
- **Feature Extraction**: Analyze URL properties like domain age, length, subdomain count, special characters, etc.
- **ML Classification**: Use an ML model to classify URLs as safe or phishing.
- **Highlight Suspicious URLs**: Display visual cues (e.g., red underline) for phishing URLs.
- **User Feedback**: Allow users to report false positives/negatives.

### 4. Development Phases
#### Phase 1: Chrome Extension Development
- Create a `manifest.json` file and popup HTML.
- Add content scripts to read webpage URLs.
#### Phase 2: Machine Learning Model
- Train a classification model using phishing datasets.
- Save the model as `.pkl` or `.onnx`.
#### Phase 3: Integration
- Connect the extension to the ML backend using APIs.
- Display results in the browser.
#### Phase 4: Testing
- Perform unit testing on ML models.
- Test extension on diverse websites.

---

## Code Implementation

### 1. Chrome Extension

#### **Manifest File (`manifest.json`)**
```json
{
  "manifest_version": 3,
  "name": "Phishing URL Detector",
  "version": "1.0",
  "description": "Detects potential phishing URLs in browser content.",
  "permissions": ["scripting", "activeTab", "storage"],
  "host_permissions": ["<all_urls>"],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"]
    }
  ],
  "action": {
    "default_popup": "popup.html",
    "default_icon": "icon.png"
  }
}
Content Script (content.js)
javascript
Copy code
document.addEventListener("DOMContentLoaded", () => {
  const urls = Array.from(document.links).map(link => link.href);

  // Send URLs to backend API
  fetch("http://127.0.0.1:5000/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ urls })
  })
    .then(response => response.json())
    .then(data => {
      // Highlight suspicious URLs
      document.querySelectorAll("a").forEach(link => {
        if (data.suspicious.includes(link.href)) {
          link.style.color = "red";
          link.title = "Potential phishing URL detected!";
        }
      });
    });
});
Popup HTML (popup.html)
html
Copy code
<!DOCTYPE html>
<html>
<head>
  <title>Phishing URL Detector</title>
</head>
<body>
  <h1>Phishing URL Detector</h1>
  <p>Monitoring active tab for malicious URLs...</p>
</body>
</html>
2. Backend
Model Training (train_model.py)
python
Copy code
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("phishing_dataset.csv")

# Feature extraction
features = data[["url_length", "has_ip", "num_subdomains", "special_chars"]]
labels = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save model
with open("phishing_model.pkl", "wb") as file:
    pickle.dump(model, file)
API Server (api.py)
python
Copy code
from flask import Flask, request, jsonify
import pickle

# Load model
with open("phishing_model.pkl", "rb") as file:
    model = pickle.load(file)

# Feature extraction function
def extract_features(url):
    import tldextract
    domain_info = tldextract.extract(url)
    return {
        "url_length": len(url),
        "has_ip": url.count("0") > 0 or url.count(".") > 2,
        "num_subdomains": len(domain_info.subdomain.split(".")),
        "special_chars": sum(1 for c in url if c in "!@#$%^&*()")
    }

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    urls = request.json.get("urls", [])
    results = {"safe": [], "suspicious": []}

    for url in urls:
        features = extract_features(url)
        prediction = model.predict([list(features.values())])[0]
        if prediction == 1:
            results["suspicious"].append(url)
        else:
            results["safe"].append(url)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
3. Dataset Example (phishing_dataset.csv)
csv
Copy code
url,url_length,has_ip,num_subdomains,special_chars,label
http://example.com,20,0,1,0,0
http://phishing.com/login.php,30,0,0,2,1
Future Enhancements
Add OAuth for authentication to report phishing URLs.
Deploy the ML model using a scalable cloud platform (e.g., AWS, Google Cloud).
Enable real-time feedback loops for improving model accuracy.

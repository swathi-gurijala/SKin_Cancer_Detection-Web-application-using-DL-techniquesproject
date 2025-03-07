import React, { useState } from "react";
import UploadComponent from "./UploadComponent"; // Import the UploadComponent
import axios from "axios";

function App() {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState("");
    const [risk, setRisk] = useState("");
    const [description, setDescription] = useState("");
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            setLoading(true); // Show loading state
            console.log("Sending request to Flask...");

            const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            console.log("Response received:", response.data);
            setPrediction(response.data.prediction);
            setRisk(response.data.risk);
            setDescription(response.data.description);
        } catch (error) {
            console.error("Error uploading image:", error.response ? error.response.data : error.message);
            setPrediction("Error: Could not get prediction.");
        } finally {
            setLoading(false); // Hide loading state
        }
    };

    return (
        <div style={{ textAlign: "center", marginTop: "50px", fontFamily: "Arial, sans-serif" }}>
            <h2 style={{ 
                border: "2px solid red", 
                display: "inline-block", 
                padding: "10px", 
                borderRadius: "5px",
                color: "#fff",
                backgroundColor: "red",
            }}>
                Skin Cancer Detection
            </h2>
            
            <div style={{ marginTop: "20px" }}>
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <br />
                <button 
                    onClick={handleUpload} 
                    style={{
                        marginTop: "10px",
                        backgroundColor: "red",
                        color: "white",
                        padding: "10px 15px",
                        border: "none",
                        cursor: "pointer",
                        borderRadius: "5px"
                    }}
                >
                    {loading ? "Processing..." : "Upload & Predict"}
                </button>
            </div>

            {prediction && (
                <div style={{
                    marginTop: "20px",
                    padding: "15px",
                    border: "1px solid #ccc",
                    backgroundColor: "#f8f8f8",
                    display: "inline-block",
                    borderRadius: "5px"
                }}>
                    <h3>Prediction: {prediction}</h3>
                    <p><strong>Risk Level: </strong>{risk}</p>
                    <p><strong>Description: </strong>{description}</p>
                </div>
            )}

            {/* Render the UploadComponent which contains doctor details */}
            <UploadComponent />
        </div>
    );
}

export default App;

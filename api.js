import axios from "axios";

const API_URL = "https://your-backend.onrender.com";

export const analyzeText = async (text) => {
    try {
        const response = await axios.post(API_URL, { text });
        return response.data;
    } catch (error) {
        console.error("Error analyzing text:", error);
        return { status: "failed", error: error.message };
    }
};

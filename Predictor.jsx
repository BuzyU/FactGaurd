// frontend/src/components/Predictor.jsx
import { useState } from "react";
import { getPrediction } from "../api";

export default function Predictor() {
    const [text, setText] = useState("");
    const [prediction, setPrediction] = useState(null);

    const handlePredict = async () => {
        const result = await getPrediction(text);
        setPrediction(result);
    };

    return (
        <div className="p-4">
            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="w-full p-2 border"
            />
            <button onClick={handlePredict} className="p-2 mt-2 text-white bg-blue-500 rounded">
                Predict
            </button>
            {prediction && <pre className="mt-4">{JSON.stringify(prediction, null, 2)}</pre>}
        </div>
    );
}

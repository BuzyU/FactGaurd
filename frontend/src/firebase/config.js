// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDAiv6xXsrZHR2uEmBlcTATkcfOigOGi0k",
  authDomain: "factguard-5f9f2.firebaseapp.com",
  projectId: "factguard-5f9f2",
  storageBucket: "factguard-5f9f2.firebasestorage.app",
  messagingSenderId: "63110462291",
  appId: "1:63110462291:web:1e797f037d2de7abefbfa6",
  measurementId: "G-YDV27RHF9H"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);

export { auth, analytics };
export default app;

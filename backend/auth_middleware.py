from fastapi import HTTPException, Header
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
        logger.info("Firebase already initialized")
    except ValueError:
        # Initialize Firebase with service account key or default credentials
        try:
            # Try to use service account key file if available
            if os.path.exists("firebase-service-account.json"):
                cred = credentials.Certificate("firebase-service-account.json")
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized with service account")
            else:
                # Use default credentials (for deployment)
                firebase_admin.initialize_app()
                logger.info("Firebase initialized with default credentials")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            # For development, we'll allow bypass
            logger.warning("Running without Firebase authentication")

# Initialize Firebase on module import
initialize_firebase()

async def verify_firebase_token(authorization: Optional[str] = Header(None)) -> str:
    """
    Verify Firebase ID token from Authorization header
    Returns user_id if valid, raises HTTPException if invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header format"
            )
        
        token = authorization.split("Bearer ")[1]
        
        # Verify the token with Firebase
        try:
            decoded_token = auth.verify_id_token(token)
            user_id = decoded_token['uid']
            logger.info(f"Authenticated user: {user_id}")
            return user_id
        except firebase_admin.auth.InvalidIdTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid Firebase token"
            )
        except firebase_admin.auth.ExpiredIdTokenError:
            raise HTTPException(
                status_code=401,
                detail="Expired Firebase token"
            )
        except Exception as e:
            logger.error(f"Firebase token verification error: {e}")
            # For development, return a mock user ID
            if "development" in os.environ.get("ENVIRONMENT", "").lower():
                logger.warning("Development mode: bypassing authentication")
                return "dev_user_123"
            raise HTTPException(
                status_code=401,
                detail="Token verification failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication service error"
        )

def create_custom_token(user_id: str) -> str:
    """
    Create a custom Firebase token for testing purposes
    Only use in development environment
    """
    try:
        custom_token = auth.create_custom_token(user_id)
        return custom_token.decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating custom token: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create authentication token"
        )

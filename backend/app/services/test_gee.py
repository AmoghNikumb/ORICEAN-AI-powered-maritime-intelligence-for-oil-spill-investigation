import os
import ee
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GEE_PROJECT_ID")

print("Using project:", PROJECT_ID)

ee.Authenticate(force=True)

ee.Initialize(project=PROJECT_ID)

print("Google Earth Engine connected successfully!")
import asyncio
import uvicorn
from dotenv import load_dotenv
from core import settings

load_dotenv()

if __name__ == "__main__":
    # Linux uses the default event loop policy which works well with async database drivers
    # No need for Windows-specific event loop policy adjustments
    uvicorn.run(
        "service:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.is_dev()
    )
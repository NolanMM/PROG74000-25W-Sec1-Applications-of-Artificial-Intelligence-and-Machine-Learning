from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import re
import logging
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
from scipy.special import softmax

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Sentiment Analysis API", version="1.0.0")

# Global variables for model
model = None
tokenizer = None

class VideoRequest(BaseModel):
    video_url: str

class SentimentResponse(BaseModel):
    NEGATIVE: float
    NEUTRAL: float
    POSITIVE: float
    video_id: str

def videoID(link):
    """Extract video ID from YouTube URL"""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", link)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube link")

def GetTranscript(video_id):
    """Get transcript for a YouTube video"""
    try:
        # Use the static method correctly
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript_list])
        return full_transcript
    except Exception as e:
        logger.error(f"Could not fetch transcript for video ID {video_id}. Error: {e}")
        return None

def perform_sentiment_analysis(text):
    """Perform sentiment analysis on text"""
    if not text:
        return {"NEGATIVE": None, "NEUTRAL": None, "POSITIVE": None}
    
    # Tokenize and run inference
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**enc)
    
    scores = softmax(out.logits[0].detach().numpy())
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    return {labels[i]: float(np.round(scores[i], 4)) for i in range(3)}

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model, tokenizer
    logger.info("Loading sentiment analysis model...")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    logger.info("Model loaded successfully!")

@app.get("/")
async def root():
    return {"message": "YouTube Sentiment Analysis API is running!", "status": "healthy"}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: VideoRequest):
    """Analyze sentiment of a YouTube video"""
    try:
        logger.info(f"Received request for URL: {request.video_url}")
        
        # Extract video ID
        video_id = videoID(request.video_url)
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get transcript
        transcript = GetTranscript(video_id)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not available for this video")
        
        logger.info(f"Transcript length: {len(transcript)} characters")
        
        # Analyze sentiment
        sentiment_scores = perform_sentiment_analysis(transcript)
        
        if sentiment_scores["NEGATIVE"] is None:
            raise HTTPException(status_code=500, detail="Failed to analyze sentiment")
        
        logger.info(f"Analysis complete: {sentiment_scores}")
        return SentimentResponse(**sentiment_scores, video_id=video_id)
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Test endpoint for debugging
@app.get("/test/{video_id}")
async def test_transcript(video_id: str):
    """Test endpoint to check if transcript can be fetched"""
    transcript = GetTranscript(video_id)
    return {
        "video_id": video_id,
        "transcript_available": transcript is not None,
        "transcript_length": len(transcript) if transcript else 0,
        "transcript_preview": transcript[:200] + "..." if transcript and len(transcript) > 200 else transcript
    }

@app.post("/demo", response_model=SentimentResponse)
async def demo_sentiment():
    """Demo endpoint with sample text to show sentiment analysis working"""
    logger.info("Demo request received")
    
    # Sample positive text for demonstration
    sample_text = """
    This is an absolutely amazing and wonderful video about technology and innovation! 
    I'm so excited and happy to share these incredible discoveries with everyone. 
    The future looks incredibly bright and promising for all of us. 
    This breakthrough will help millions of people around the world.
    Thank you so much for watching and I hope you enjoyed this fantastic content!
    """
    
    # Perform sentiment analysis on sample text
    sentiment_scores = perform_sentiment_analysis(sample_text)
    
    logger.info(f"Demo analysis complete: {sentiment_scores}")
    return SentimentResponse(**sentiment_scores, video_id="demo-positive-sample")

@app.post("/demo-negative", response_model=SentimentResponse) 
async def demo_sentiment_negative():
    """Demo endpoint with negative sample text"""
    logger.info("Demo negative request received")
    
    # Sample negative text for demonstration
    sample_text = """
    This is absolutely terrible and awful content. I hate everything about this video.
    This is the worst thing I've ever seen and it makes me angry and frustrated.
    Nothing works properly and everything is broken and disappointing.
    I'm completely dissatisfied and unhappy with this terrible experience.
    """
    
    sentiment_scores = perform_sentiment_analysis(sample_text)
    
    logger.info(f"Demo negative analysis complete: {sentiment_scores}")
    return SentimentResponse(**sentiment_scores, video_id="demo-negative-sample")

# Confession Moderation Feature

## Overview
This update introduces automated content moderation for confessions using the Google Gemini 2.5 Flash model. The system analyzes confession content before publishing to ensure it adheres to platform guidelines.

## New Features
- Automated confession moderation using Google's Gemini 2.5 Flash AI model
- Real-time content analysis for policy violations
- Detailed rejection reasons for users when content is flagged
- Fallback mechanisms to ensure system resilience

## Technical Changes
- Implemented `analyze_confession_with_llm` function in `main.py` to interface with the Gemini API
- Modified the `/confessions` endpoint to incorporate moderation before saving content
- Added caching for moderation results to improve performance
- Implemented comprehensive error handling and logging

## Setup Requirements
To use this feature, you must:

1. Obtain a Google Gemini API key from the [Google AI Studio](https://ai.google.dev/)
2. Add the following to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Moderation Criteria
The system rejects content that:
- Contains hate speech, explicit threats, or harassment
- Reveals private information about specific individuals without consent
- Promotes illegal activities or self-harm
- Contains sexually explicit content or graphic violence

## Fallback Behavior
If the moderation service is unavailable (API key not set or service error):
- The system will default to approving confessions
- A warning will be logged for monitoring purposes
- The platform will continue to function without interruption

## Future Improvements
- Customizable moderation policies
- Admin dashboard for moderation statistics and manual review
- User feedback mechanism for rejected confessions

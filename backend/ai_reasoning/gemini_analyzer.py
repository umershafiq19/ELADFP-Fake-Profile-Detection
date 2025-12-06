import google.generativeai as genai
import os

class GeminiAnalyzer:
    """Generate AI reasoning for predictions"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 150,
                'top_p': 0.8,
                'top_k': 40
            }
        )
    
    def generate_reasoning(self, features, prediction, confidence):
        """Generate AI analysis of the prediction"""
        try:
            # Build feature analysis
            feature_analysis = []
            
            if features.get('profile pic') == 1:
                feature_analysis.append("Has profile picture")
            else:
                feature_analysis.append("No profile picture")
            
            username_nums = features.get('nums/length username', 0)
            if username_nums > 0.5:
                feature_analysis.append(f"Username contains {int(username_nums * 100)}% numbers (bot-like)")
            else:
                feature_analysis.append(f"Username contains few numbers ({int(username_nums * 100)}%)")
            
            if features.get('name==username') == 1:
                feature_analysis.append("Name equals username (suspicious)")
            else:
                feature_analysis.append("Name differs from username")
            
            bio_length = features.get('description length', 0)
            if bio_length == 0:
                feature_analysis.append("No bio/description")
            elif bio_length < 20:
                feature_analysis.append(f"Short bio ({bio_length} chars)")
            else:
                feature_analysis.append(f"Detailed bio ({bio_length} chars)")
            
            if features.get('external URL') == 1:
                feature_analysis.append("Has external website")
            else:
                feature_analysis.append("No external website")
            
            followers = features.get('#followers', 0)
            following = features.get('#following', 0)
            
            if followers > 0:
                ratio = following / (followers + 1)
                if ratio > 5:
                    feature_analysis.append(f"High following/follower ratio ({ratio:.1f}:1)")
                elif ratio < 0.2:
                    feature_analysis.append(f"Low following/follower ratio ({ratio:.1f}:1)")
                else:
                    feature_analysis.append(f"Balanced follower ratio ({ratio:.1f}:1)")
            
            posts = features.get('#posts', 0)
            if posts == 0:
                feature_analysis.append("❌ No posts")
            elif posts < 5:
                feature_analysis.append(f"Few posts ({posts})")
            else:
                feature_analysis.append(f"Active account ({posts} posts)")
            
            # Build prompt
            prompt = f"""
Analyze this Instagram profile briefly.

Result: {"FAKE" if prediction == 1 else "REAL"} ({confidence['fake_profile_prob'] * 100:.0f}% fake)
Profile: {features.get('#posts', 0)} posts, {features.get('#followers', 0)} followers, {features.get('#following', 0)} following
Username: {int(features.get('nums/length username', 0) * 100)}% numbers, Bio: {features.get('description length', 0)} chars
Picture: {"Yes" if features.get('profile pic') == 1 else "No"}, Link: {"Yes" if features.get('external URL') == 1 else "No"}

Explain WHY it's {"fake" if prediction == 1 else "real"} based on these metrics.
"""
            
            response = self.model.generate_content(prompt, request_options={'timeout': 10})
            return response.text.strip()
        
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return (
                "Fake profile indicators: incomplete bio, unusual ratios, number-heavy usernames."
                if prediction == 1 else
                "Real profile indicators: detailed bio, normal ratios, natural username behavior."
            )
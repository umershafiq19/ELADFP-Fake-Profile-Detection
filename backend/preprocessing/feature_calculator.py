import numpy as np

class FeatureCalculator:
    """Calculate derived features from raw profile data"""
    
    @staticmethod
    def calculate_nums_ratio(text):
        """Calculate ratio of numbers to total length"""
        if len(text) == 0:
            return 0.0
        nums_count = sum(1 for char in text if char.isdigit())
        return round(nums_count / len(text), 3)
    
    @staticmethod
    def count_words(text):
        """Count number of words in text"""
        if not text or len(text.strip()) == 0:
            return 0
        return len(text.strip().split())
    
    @staticmethod
    def check_name_equals_username(username, fullname):
        """Check if fullname matches username"""
        # Remove spaces from fullname and convert to lowercase
        clean_fullname = fullname.lower().replace(' ', '')
        clean_username = username.lower()
        return 1 if clean_fullname == clean_username else 0
    
    @classmethod
    def preprocess_features(cls, input_data):
        """
        Transform raw input into model features
        
        Args:
            input_data: dict with raw features
            
        Returns:
            dict with all preprocessed features
        """
        username = input_data.get('username', '')
        fullname = input_data.get('fullname', '')
        bio = input_data.get('bio', '')
        
        # Calculate derived features
        processed = {
            'profile pic': int(input_data.get('profile pic', 0)),
            'nums/length username': cls.calculate_nums_ratio(username),
            'fullname words': cls.count_words(fullname),
            'nums/length fullname': cls.calculate_nums_ratio(fullname),
            'name==username': cls.check_name_equals_username(username, fullname),
            'description length': len(bio),
            'external URL': int(input_data.get('external URL', 0)),
            'private': int(input_data.get('private', 0)),
            '#posts': int(input_data.get('#posts', 0)),
            '#followers': int(input_data.get('#followers', 0)),
            '#following': int(input_data.get('#following', 0))
        }
        
        return processed
    
    @staticmethod
    def apply_log_transforms(df):
        """Apply log transformations to engagement metrics"""
        import numpy as np
        
        df['log_followers'] = np.log1p(df['#followers'])
        df['log_following'] = np.log1p(df['#following'])
        df['log_posts'] = np.log1p(df['#posts'])
        
        # Calculate ratios
        df['followers_to_following'] = df['#followers'] / (df['#following'] + 1)
        df['posts_per_follower'] = df['#posts'] / (df['#followers'] + 1)
        
        # Clip extreme values
        df['followers_to_following'] = np.clip(df['followers_to_following'], 0, 10)
        df['posts_per_follower'] = np.clip(df['posts_per_follower'], 0, 5)
        
        # Drop raw columns
        df = df.drop(columns=['#followers', '#following', '#posts'])
        
        return df
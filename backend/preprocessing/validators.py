class InputValidator:
    """Validate user input"""
    
    REQUIRED_FIELDS = [
        'profile pic', 'nums/length username', 'fullname words', 
        'nums/length fullname', 'name==username', 'description length',
        'external URL', 'private', '#posts', '#followers', '#following'
    ]
    
    @staticmethod
    def validate_binary_field(value, field_name):
        """Validate binary fields (0 or 1)"""
        try:
            val = int(value)
            if val not in [0, 1]:
                raise ValueError(f"{field_name} must be 0 or 1")
            return val
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a valid integer (0 or 1)")
    
    @staticmethod
    def validate_numeric_field(value, field_name, min_val=0):
        """Validate numeric fields"""
        try:
            val = float(value)
            if val < min_val:
                raise ValueError(f"{field_name} must be >= {min_val}")
            return val
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a valid number")
    
    @classmethod
    def validate_input(cls, data):
        """Validate all input fields"""
        errors = []
        
        # Check required fields
        missing = [f for f in cls.REQUIRED_FIELDS if f not in data]
        if missing:
            errors.append(f"Missing fields: {missing}")
        
        # Validate binary fields
        binary_fields = ['profile pic', 'name==username', 'external URL', 'private']
        for field in binary_fields:
            if field in data:
                try:
                    cls.validate_binary_field(data[field], field)
                except ValueError as e:
                    errors.append(str(e))
        
        # Validate numeric fields
        numeric_fields = ['#posts', '#followers', '#following']
        for field in numeric_fields:
            if field in data:
                try:
                    cls.validate_numeric_field(data[field], field, min_val=0)
                except ValueError as e:
                    errors.append(str(e))
        
        if errors:
            return False, errors
        return True, []
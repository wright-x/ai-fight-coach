    def create_user(self, email: str, name: str = None) -> Optional[User]:
        """Create a new user or return existing one"""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(User.email == email).first()
            if existing_user:
                logger.info(f"User already exists: {existing_user.id}")
                return existing_user  # Return existing user instead of None
            
            # Create new user (without name column)
            user = User(email=email)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"Created new user: {user.id}")
            return user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            self.db.rollback()
            return None
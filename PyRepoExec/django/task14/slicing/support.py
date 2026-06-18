from django.contrib.auth.tokens import PasswordResetTokenGenerator

class FixedNowPasswordResetTokenGenerator(PasswordResetTokenGenerator):
    def __init__(self, now):
        super().__init__()
        self._fixed_now = now

    def _now(self):
        return self._fixed_now

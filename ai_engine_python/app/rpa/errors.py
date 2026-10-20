"""Application-owned failures whose messages can safely appear in task logs."""


class RpaIssue(ValueError):
    def __init__(self, code, message, evidence=None):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.evidence = evidence

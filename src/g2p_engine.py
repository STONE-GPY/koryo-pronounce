from typing import List
from g2pk import G2p

class KoryoG2PEngine:
    """Koryo-saram focused Korean G2P (Grapheme-to-Phoneme) Engine.
    
    Transforms Korean text into standard phonetic representations.
    """
    
    def __init__(self) -> None:
        """Initialize the G2P model."""
        self.g2p = G2p()

    def convert(self, text: str) -> str:
        """Converts Korean text to its standard phonetic string representation.
        
        Args:
            text (str): Input Korean text.
            
        Returns:
            str: Phonetic representation (e.g., "국물" -> "궁물").
            Returns an empty string if the input is empty or None.
        """
        if not text or not isinstance(text, str):
            return ""
        
        return self.g2p(text)

    def get_phoneme_list(self, text: str) -> List[str]:
        """Returns a list of phonetic words.
        
        Args:
            text (str): Input Korean text.
            
        Returns:
            List[str]: List of phonetic words.
        """
        phonetic_text = self.convert(text)
        if not phonetic_text:
            return []
        
        return phonetic_text.split()

    def get_phonemes(self, text: str) -> List[str]:
        """Backward compatibility for get_phonemes, returning a list of phonetic words.
        
        Deprecated: Use get_phoneme_list instead.
        """
        return self.get_phoneme_list(text)

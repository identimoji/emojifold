"""
Unicode Emoji Database Generator

Generates a complete emoji table with codepoints as IDs from official Unicode ranges.
"""

import unicodedata
import json
from typing import List, Dict, Optional


class EmojiGenerator:
    """Generate comprehensive emoji database from Unicode ranges"""
    
    # Official Unicode emoji ranges
    EMOJI_RANGES = [
        # Basic Latin supplement and symbols
        (0x00A0, 0x00FF, "Latin-1 Supplement"),
        (0x2600, 0x26FF, "Miscellaneous Symbols"),
        (0x2700, 0x27BF, "Dingbats"),
        
        # Main emoji blocks
        (0x1F300, 0x1F5FF, "Miscellaneous Symbols and Pictographs"),
        (0x1F600, 0x1F64F, "Emoticons"),
        (0x1F680, 0x1F6FF, "Transport and Map Symbols"), 
        (0x1F700, 0x1F77F, "Alchemical Symbols"),
        (0x1F780, 0x1F7FF, "Geometric Shapes Extended"),
        (0x1F800, 0x1F8FF, "Supplemental Arrows-C"),
        (0x1F900, 0x1F9FF, "Supplemental Symbols and Pictographs"),
        
        # Extended emoji
        (0x1FA00, 0x1FA6F, "Chess Symbols"),
        (0x1FA70, 0x1FAFF, "Symbols and Pictographs Extended-A"),
    ]
    
    def __init__(self):
        self.emojis = []
        self.categories = {}
    
    def is_emoji(self, char: str) -> bool:
        """Check if character is actually an emoji"""
        try:
            # Check if it's a printable character
            if not char.isprintable():
                return False
                
            # Get Unicode name
            name = unicodedata.name(char, "").upper()
            
            # Skip if no name or control character
            if not name or "CONTROL" in name:
                return False
                
            # Check for emoji indicators in name
            emoji_indicators = [
                "FACE", "SMILE", "GRIN", "HEART", "STAR", "SUN", "MOON",
                "FIRE", "WATER", "TREE", "FLOWER", "ANIMAL", "CAT", "DOG", 
                "BIRD", "FISH", "BUG", "SNAKE", "LION", "TIGER", "BEAR",
                "RABBIT", "MOUSE", "COW", "PIG", "FROG", "MONKEY",
                "ROCKET", "CAR", "PLANE", "SHIP", "HOUSE", "FOOD",
                "FRUIT", "VEGETABLE", "DRINK", "SPORT", "BALL", "GAME"
            ]
            
            # Additional check for specific ranges known to contain emoji
            codepoint = ord(char)
            if (0x1F600 <= codepoint <= 0x1F64F or  # Emoticons
                0x1F300 <= codepoint <= 0x1F5FF or  # Misc Symbols and Pictographs
                0x1F680 <= codepoint <= 0x1F6FF or  # Transport and Map
                0x1F900 <= codepoint <= 0x1F9FF):   # Supplemental Symbols
                return True
                
            # Check if name contains emoji indicators
            return any(indicator in name for indicator in emoji_indicators)
            
        except (ValueError, TypeError):
            return False
    
    def get_emoji_category(self, char: str, codepoint: int) -> str:
        """Determine emoji category based on codepoint and name"""
        try:
            name = unicodedata.name(char, "").upper()
            
            # Category based on Unicode block
            if 0x1F600 <= codepoint <= 0x1F64F:
                return "face"
            elif 0x1F680 <= codepoint <= 0x1F6FF:
                return "transport"
            elif 0x1F900 <= codepoint <= 0x1F9FF:
                return "supplemental"
            elif 0x2600 <= codepoint <= 0x26FF:
                return "symbol"
            elif 0x1F300 <= codepoint <= 0x1F5FF:
                # Subcategorize the large misc block
                if any(word in name for word in ["FACE", "SMILE", "GRIN"]):
                    return "face"
                elif any(word in name for word in ["ANIMAL", "CAT", "DOG", "BIRD", "FISH", "BUG", "SNAKE", "LION", "TIGER", "BEAR", "RABBIT", "MOUSE", "COW", "PIG", "FROG", "MONKEY"]):
                    return "animal"
                elif any(word in name for word in ["SUN", "MOON", "STAR", "FIRE", "WATER", "TREE", "FLOWER"]):
                    return "nature"
                elif any(word in name for word in ["HEART", "CIRCLE", "SQUARE", "DIAMOND"]):
                    return "symbol"
                else:
                    return "misc"
            else:
                return "other"
        except:
            return "unknown"
    
    def generate_emoji_list(self) -> List[Dict]:
        """Generate comprehensive emoji list from Unicode ranges"""
        emoji_list = []
        
        for start, end, block_name in self.EMOJI_RANGES:
            print(f"Processing {block_name}: U+{start:04X}-U+{end:04X}")
            
            for codepoint in range(start, end + 1):
                try:
                    char = chr(codepoint)
                    
                    # Skip if not actually an emoji
                    if not self.is_emoji(char):
                        continue
                    
                    # Get Unicode name and category
                    name = unicodedata.name(char, f"U+{codepoint:04X}")
                    category = self.get_emoji_category(char, codepoint)
                    
                    emoji_data = {
                        "codepoint": codepoint,
                        "emoji": char,
                        "name": name,
                        "category": category,
                        "block": block_name,
                        "hex": f"U+{codepoint:04X}"
                    }
                    
                    emoji_list.append(emoji_data)
                    
                except (ValueError, UnicodeDecodeError):
                    # Skip invalid codepoints
                    continue
        
        return sorted(emoji_list, key=lambda x: x["codepoint"])
    
    def generate_sql_insert(self, emoji_list: List[Dict]) -> str:
        """Generate SQL INSERT statements for emoji table"""
        sql_lines = [
            "-- Complete Unicode Emoji Database",
            "-- Generated from official Unicode ranges",
            "",
            "-- Create emojis table with codepoint as ID",
            "CREATE TABLE IF NOT EXISTS emojis (",
            "    codepoint INTEGER PRIMARY KEY,",
            "    emoji TEXT NOT NULL,",
            "    name TEXT NOT NULL,",
            "    category TEXT NOT NULL,",
            "    unicode_block TEXT NOT NULL,",
            "    hex_code TEXT NOT NULL,",
            "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            ");",
            "",
            "-- Insert all emojis",
            "INSERT OR IGNORE INTO emojis (codepoint, emoji, name, category, unicode_block, hex_code) VALUES"
        ]
        
        # Generate INSERT values
        values = []
        for emoji in emoji_list:
            # Escape single quotes in emoji name for SQL
            escaped_name = emoji['name'].replace("'", "''")
            escaped_block = emoji['block'].replace("'", "''")
            
            values.append(
                f"({emoji['codepoint']}, '{emoji['emoji']}', "
                f"'{escaped_name}', '{emoji['category']}', "
                f"'{escaped_block}', '{emoji['hex']}')"
            )
        
        sql_lines.append(",\n".join(values) + ";")
        
        # Add indexes
        sql_lines.extend([
            "",
            "-- Indexes for performance",
            "CREATE INDEX IF NOT EXISTS idx_emojis_category ON emojis(category);",
            "CREATE INDEX IF NOT EXISTS idx_emojis_block ON emojis(unicode_block);"
        ])
        
        return "\n".join(sql_lines)


def main():
    """Generate emoji database"""
    generator = EmojiGenerator()
    
    print("🔍 Generating comprehensive emoji database...")
    emoji_list = generator.generate_emoji_list()
    
    print(f"✅ Found {len(emoji_list)} emojis")
    
    # Show category breakdown
    categories = {}
    for emoji in emoji_list:
        cat = emoji["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Category breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Generate SQL
    sql = generator.generate_sql_insert(emoji_list)
    
    # Save to file
    with open("unicode_emojis.sql", "w", encoding="utf-8") as f:
        f.write(sql)
    
    # Save JSON for reference
    with open("unicode_emojis.json", "w", encoding="utf-8") as f:
        json.dump(emoji_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to:")
    print(f"  unicode_emojis.sql ({len(emoji_list)} emoji INSERT statements)")
    print(f"  unicode_emojis.json (JSON reference)")


if __name__ == "__main__":
    main()

"""
Generate Synthetic PII for MovieLens Users (MAMA Integration)

This tool generates synthetic Personally Identifiable Information (PII) for
MovieLens users following the MAMA paper's 5 PII categories:
1. Identity (name, username)
2. Contact (email, phone)
3. Location (address, country, zipcode)
4. Temporal (date-of-birth, timestamps)
5. Regulated Identifiers (SSN, unique IDs)

It also generates pseudonyms and creates an enhanced u.user file.

Usage:
    python connacf/tools/generate_pii_for_movielens.py \\
        --input dataset/ml-100k-raw/ml-100k/u.user \\
        --output dataset/ml-100k/ml-100k.user \\
        --seed 42
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple


class PIIGenerator:
    """Generate synthetic PII following MAMA taxonomy."""
    
    # First names for pseudonyms (diverse, gender-neutral)
    FIRST_NAMES = [
        "Alex", "Blake", "Casey", "Drew", "Ellis", "Finley", "Gray", "Harper",
        "Indigo", "Jordan", "Kai", "Logan", "Morgan", "Noel", "Oakley", "Parker",
        "Quinn", "Reese", "Sage", "Taylor"
    ]
    
    # Last names for pseudonyms
    LAST_NAMES = [
        "Anderson", "Brooks", "Chen", "Davis", "Evans", "Foster", "Garcia", "Harris",
        "Ivanov", "Johnson", "Kim", "Lee", "Martinez", "Nguyen", "O'Brien", "Patel",
        "Quinn", "Rodriguez", "Smith", "Taylor", "Ueda", "Vasquez", "Wang", "Xavier",
        "Yang", "Zhang"
    ]
    
    # Email domains
    EMAIL_DOMAINS = [
        "gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "icloud.com",
        "mail.com", "aol.com", "zoho.com"
    ]
    
    # US States for addresses (full names to avoid false positive matching)
    US_STATES = [
        "California", "New York", "Texas", "Florida", "Illinois", 
        "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan",
        "New Jersey", "Virginia", "Washington", "Arizona", "Massachusetts", 
        "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin"
    ]
    
    # Street names
    STREET_NAMES = [
        "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Park",
        "Lake", "Hill", "River", "Forest", "Meadow", "Spring", "Valley"
    ]
    
    STREET_TYPES = ["St", "Ave", "Rd", "Blvd", "Dr", "Ln", "Way", "Ct"]
    
    def __init__(self, seed: int = 42):
        """Initialize PII generator with seed for reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)
        
    def generate_name(self, user_id: int) -> Tuple[str, str, str]:
        """
        Generate pseudonym (first, last, full).
        
        Returns:
            (first_name, last_name, full_name)
        """
        # Use different prime multipliers to ensure diversity
        first_idx = user_id % len(self.FIRST_NAMES)
        # Use a prime multiplier (17) to spread out last names
        last_idx = (user_id * 17) % len(self.LAST_NAMES)
        
        first = self.FIRST_NAMES[first_idx]
        last = self.LAST_NAMES[last_idx]
        full = f"{first} {last}"
        
        return first, last, full
    
    def generate_email(self, first_name: str, last_name: str, user_id: int) -> str:
        """Generate email address."""
        domain_idx = user_id % len(self.EMAIL_DOMAINS)
        domain = self.EMAIL_DOMAINS[domain_idx]
        
        # Various email formats
        formats = [
            f"{first_name.lower()}.{last_name.lower()}@{domain}",
            f"{first_name.lower()}{last_name.lower()}@{domain}",
            f"{first_name[0].lower()}{last_name.lower()}@{domain}",
            f"{first_name.lower()}{user_id}@{domain}",
        ]
        
        format_idx = user_id % len(formats)
        return formats[format_idx]
    
    def generate_phone(self, user_id: int) -> str:
        """Generate US phone number."""
        # Area code (avoid 555 for realism)
        area_codes = [201, 212, 213, 214, 303, 310, 312, 404, 415, 510, 617, 650, 702, 713, 818, 917]
        area_code = area_codes[user_id % len(area_codes)]
        
        # Exchange and subscriber
        exchange = 200 + (user_id * 7) % 800
        subscriber = 1000 + (user_id * 13) % 9000
        
        return f"{area_code}-{exchange:03d}-{subscriber:04d}"
    
    def generate_address(self, user_id: int, zipcode: str) -> str:
        """Generate street address."""
        street_num = 100 + (user_id * 17) % 9900
        street_name_idx = user_id % len(self.STREET_NAMES)
        street_type_idx = (user_id // len(self.STREET_NAMES)) % len(self.STREET_TYPES)
        
        street_name = self.STREET_NAMES[street_name_idx]
        street_type = self.STREET_TYPES[street_type_idx]
        
        return f"{street_num} {street_name} {street_type}"
    
    def generate_city_state(self, zipcode: str) -> Tuple[str, str]:
        """Generate city and state from zipcode."""
        # Simple mapping based on zipcode prefix
        zip_prefix = int(zipcode[:2]) if zipcode and len(zipcode) >= 2 and zipcode[:2].isdigit() else 0
        state_idx = zip_prefix % len(self.US_STATES)
        state = self.US_STATES[state_idx]
        
        # Generate city name (expanded mappings for all states)
        cities = {
            "California": ["Los Angeles", "San Francisco", "San Diego", "Sacramento"],
            "New York": ["New York City", "Buffalo", "Rochester", "Albany"],
            "Texas": ["Houston", "Dallas", "Austin", "San Antonio"],
            "Florida": ["Miami", "Tampa", "Orlando", "Jacksonville"],
            "Illinois": ["Chicago", "Aurora", "Naperville", "Rockford"],
            "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie"],
            "Ohio": ["Columbus", "Cleveland", "Cincinnati", "Toledo"],
            "Georgia": ["Atlanta", "Augusta", "Savannah", "Athens"],
            "North Carolina": ["Charlotte", "Raleigh", "Durham", "Greensboro"],
            "Michigan": ["Detroit", "Grand Rapids", "Ann Arbor", "Lansing"],
            "New Jersey": ["Newark", "Jersey City", "Trenton", "Princeton"],
            "Virginia": ["Virginia Beach", "Richmond", "Norfolk", "Arlington"],
            "Washington": ["Seattle", "Spokane", "Tacoma", "Bellevue"],
            "Arizona": ["Phoenix", "Tucson", "Mesa", "Scottsdale"],
            "Massachusetts": ["Boston", "Cambridge", "Worcester", "Springfield"],
            "Tennessee": ["Nashville", "Memphis", "Knoxville", "Chattanooga"],
            "Indiana": ["Indianapolis", "Fort Wayne", "Bloomington", "Evansville"],
            "Missouri": ["Kansas City", "St. Louis", "Springfield", "Columbia"],
            "Maryland": ["Baltimore", "Annapolis", "Rockville", "Frederick"],
            "Wisconsin": ["Milwaukee", "Madison", "Green Bay", "Kenosha"],
        }
        
        default_cities = ["Springfield", "Riverside", "Fairview", "Clinton"]
        city_list = cities.get(state, default_cities)
        city_idx = (zip_prefix // 10) % len(city_list)
        city = city_list[city_idx]
        
        return city, state
    
    def generate_dob(self, age: int, user_id: int) -> str:
        """Generate date of birth from age."""
        # Assume current year is 1998 (MovieLens 100k was collected in 1998)
        birth_year = 1998 - int(age)
        
        # Generate month and day deterministically
        month = 1 + (user_id % 12)
        day = 1 + (user_id % 28)  # Safe for all months
        
        return f"{birth_year}-{month:02d}-{day:02d}"
    
    def generate_unique_id(self, user_id: int) -> str:
        """Generate unique identifier (like UID in MAMA)."""
        # Format: UID-XXXXXXXX (8 alphanumeric chars)
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # Deterministic generation
        uid_chars = []
        val = user_id
        for _ in range(8):
            uid_chars.append(chars[val % len(chars)])
            val = val // len(chars) + 7  # Add prime to mix
        
        return f"UID-{''.join(uid_chars)}"
    
    def generate_ssn(self, user_id: int) -> str:
        """Generate synthetic SSN (clearly marked as synthetic)."""
        # Format: XXX-XX-XXXX
        # Use 900-999 range which is reserved/invalid for real SSNs
        area = 900 + (user_id % 100)
        group = (user_id * 7) % 100
        serial = 1000 + (user_id * 13) % 9000
        
        return f"{area}-{group:02d}-{serial:04d}"
    
    def generate_all_pii(
        self,
        user_id: int,
        age: str,
        gender: str,
        occupation: str,
        zipcode: str
    ) -> Dict[str, str]:
        """
        Generate all PII for a user following MAMA's 5 categories.
        
        Returns:
            Dict with all PII fields
        """
        # 1. Identity
        first_name, last_name, full_name = self.generate_name(user_id)
        
        # 2. Contact
        email = self.generate_email(first_name, last_name, user_id)
        phone = self.generate_phone(user_id)
        
        # 3. Location
        address = self.generate_address(user_id, zipcode)
        city, state = self.generate_city_state(zipcode)
        
        # 4. Temporal
        dob = self.generate_dob(age, user_id)
        
        # 5. Regulated Identifiers
        unique_id = self.generate_unique_id(user_id)
        ssn = self.generate_ssn(user_id)
        
        return {
            # Identity
            'pseudonym_first': first_name,
            'pseudonym_last': last_name,
            'pseudonym_full': full_name,
            
            # Contact
            'email': email,
            'phone': phone,
            
            # Location
            'address': address,
            'city': city,
            'state': state,
            'zipcode': zipcode,
            
            # Temporal
            'age': age,
            'dob': dob,
            
            # Regulated Identifiers
            'unique_id': unique_id,
            'ssn': ssn,
            
            # Original fields
            'gender': gender,
            'occupation': occupation,
        }


def process_movielens_user_file(
    input_path: str,
    output_path: str,
    seed: int = 42
):
    """
    Process MovieLens u.user file and add synthetic PII + pseudonyms.
    
    Input format:  user_id|age|gender|occupation|zipcode
    Output format: user_id|age|gender|occupation|zipcode|pseudonym|email|phone|address|city|state|dob|unique_id|ssn
    """
    generator = PIIGenerator(seed=seed)
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Write header
        header = "user_id\tage\tgender\toccupation\tzipcode\tpseudonym\temail\tphone\taddress\tcity\tstate\tdob\tunique_id\tssn\n"
        f_out.write(header)
        
        line_count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Parse original format
            parts = line.split('|')
            if len(parts) != 5:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            user_id, age, gender, occupation, zipcode = parts
            
            # Generate all PII
            pii = generator.generate_all_pii(
                user_id=int(user_id),
                age=age,
                gender=gender,
                occupation=occupation,
                zipcode=zipcode
            )
            
            # Write enhanced line
            output_line = (
                f"{user_id}\t{age}\t{gender}\t{occupation}\t{zipcode}\t"
                f"{pii['pseudonym_full']}\t{pii['email']}\t{pii['phone']}\t"
                f"{pii['address']}\t{pii['city']}\t{pii['state']}\t"
                f"{pii['dob']}\t{pii['unique_id']}\t{pii['ssn']}\n"
            )
            f_out.write(output_line)
            
            line_count += 1
            if line_count % 100 == 0:
                print(f"Processed {line_count} users...")
    
    print(f"\n✓ Successfully processed {line_count} users")
    print(f"✓ Output written to: {output_file}")
    print(f"\nPII Categories Added (following MAMA):")
    print("  1. Identity: pseudonym (name)")
    print("  2. Contact: email, phone")
    print("  3. Location: address, city, state, zipcode")
    print("  4. Temporal: age, date-of-birth")
    print("  5. Regulated Identifiers: unique_id, ssn")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic PII for MovieLens users (MAMA integration)"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='dataset/ml-100k-raw/ml-100k/u.user',
        help='Input u.user file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset/ml-100k/ml-100k.user',
        help='Output file path with PII'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    process_movielens_user_file(
        input_path=args.input,
        output_path=args.output,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

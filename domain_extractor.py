import re
import time
import json
import math
from collections import Counter
import pandas as pd
#from scipy.stats import entropy
import jellyfish

class DomainFeatureExtractor:
    def __init__(self, whitelist):
        self.whitelist = whitelist
        self.whitelist_tokens = self.tokenize_all(whitelist)
        self.token_counts = Counter([t for sub in self.whitelist_tokens for t in sub])
        self.safe_tokens = set([token for token, _ in self.token_counts.most_common(100000)])

    def set_whitelist(self, whitelist):
        self.__init__(whitelist)

    def tokenize_domain(self, domain):
        return re.split(r'[\.\-/_]', domain.lower())

    def tokenize_all(self, domains):
        return [self.tokenize_domain(d) for d in domains if isinstance(d, str)]

    def calc_entropy(self, domain):
        counts = Counter(domain)
        probs = [count / len(domain) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def suspicious_keyword_present(self, domain):
        keywords = ['secure', 'login', 'update', 'account', 'verify', 'apple', 'bank', 'gmail', 'yahoo', 'paypal']
        return int(any(kw in domain.lower() for kw in keywords))

    def suspicious_tld(self, domain):
        bad_tlds = ['xyz', 'top', 'loan', 'click', 'gq', 'ml', 'cf', 'tk']
        return int(domain.split('.')[-1] in bad_tlds)

    def has_unusual_token(self, domain):
        tokens = set(self.tokenize_domain(domain))
        return int(any(token not in self.safe_tokens for token in tokens))

    def max_jaro_winkler_similarity(self, domain):
        return max(jellyfish.jaro_winkler_similarity(domain, ref) for ref in self.whitelist)

    def clean_domain(self, domain):
        if not isinstance(domain, str):
            return ""
        domain = domain.replace("*.", "")
        return re.sub(r"^www\.", "", domain)
    
    def has_inner_tld(self, domain):
        tlds = ['.com', '.net', '.org']
        return int(any(tld in domain[:-len(tld)] for tld in tlds))

    def count_hyphens(self, domain):
        return domain.count('-')

    def count_subdomains(self, domain):
        return len(domain.split('.')) - 2 if '.' in domain else 0
    
    def is_deeply_nested(self,domain, threshold=4):
        return int(self.count_subdomains(domain) >= threshold)

    def extract(self, df):
        df['clean_domain'] = df['parsed_domainname'].apply(self.clean_domain)
        df["not_before"] = pd.to_datetime(df["not_before"])
        df["not_after"] = pd.to_datetime(df["not_after"])
        df['validity_days'] = (df['not_after'] - df['not_before']).dt.days
        df['domain_length'] = df['clean_domain'].apply(lambda x: len(str(x)))
        df['deeply_nested_subdomains'] = df['clean_domain'].apply(self.is_deeply_nested)
        df['num_subdomains'] = df['clean_domain'].apply(self.count_subdomains)
        df['hyphen_count'] = df['clean_domain'].apply(self.count_hyphens)
        df['shannon_entropy'] = df['clean_domain'].apply(self.calc_entropy)
        df['suspicious_keyword'] = df['clean_domain'].apply(self.suspicious_keyword_present)
        df['suspicious_tld'] = df['clean_domain'].apply(self.suspicious_tld)
        df['has_inner_tld'] = df['clean_domain'].apply(self.has_inner_tld)
        df['issued_hour_utc'] = df['not_before'].dt.hour
        df['is_weekend'] = df['not_before'].dt.dayofweek >= 5
        df['has_unusual_token'] = df['clean_domain'].apply(self.has_unusual_token)


        df["creation_date"] = pd.to_datetime(df["creation_date"], errors='coerce', utc=True)
        df['domain_age_days'] = (df['not_before'] - df['creation_date']).dt.days
        target_cas = [
            'Buypass',
            'DigiCert',
            'GlobalSign nv-sa',
            'GoDaddy',
            'Google Trust Services LLC',
            'Internet Security Research Group',
            'SSL.com',
            'Sectigo'
        ]
        df['ca_name'] = df['ca_name'].where(df['ca_name'].isin(target_cas), 'Other')
        df = pd.get_dummies(df, columns=['ca_name'], drop_first=True)
        expected_ca_columns = [f'ca_name_{name}' for name in target_cas]
        for col in expected_ca_columns:
            if col not in df.columns:
                df[col] = 0
        df['max_jaro_winkler_similarity'] = df['clean_domain'].apply(self.max_jaro_winkler_similarity)
        selected_features = [

            'parsed_domainname',
            'no_domains',
            'validity_days',
            'domain_length',
            'deeply_nested_subdomains',
            'num_subdomains',
            'hyphen_count',
            'shannon_entropy',
            'suspicious_keyword',
            'suspicious_tld',
            'has_inner_tld',
            'issued_hour_utc',
            'is_weekend',
            'has_unusual_token',
            'ca_name_Buypass',
            'ca_name_DigiCert',
            'ca_name_GlobalSign nv-sa',
            'ca_name_GoDaddy',
            'ca_name_Google Trust Services LLC',
            'ca_name_Internet Security Research Group',
            'ca_name_SSL.com',
            'ca_name_Sectigo',
            'max_jaro_winkler_similarity',
            'domain_age_days'
            
        ]        
        df = df.loc[:, selected_features]
        return df
import json
import warnings
from sentence_transformers import SentenceTransformer, util

warnings.simplefilter(action='ignore', category=FutureWarning)

# Name of the input files
experts_file = 'data/experts.json'
bussiness_file = 'data/bussiness.json'

# Load JSON data from the specified file
with open(experts_file, 'r', encoding='utf-8') as file:
    experts = json.load(file)

with open(bussiness_file, 'r', encoding='utf-8') as file:
    bussiness = json.load(file)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

business_description = bussiness["businessDescription"]
business_needs = bussiness['needs']

all_needs_combined = ' '.join([need for need in business_needs.values() if need])
business_combined_text = business_description + ' ' + all_needs_combined

business_combined_embedding = model.encode(business_combined_text, convert_to_tensor=True)


def extract_meaningful_info(expert):
    meaningful_info = []

    # Extract the "about" section
    if 'about' in expert and expert['about']:
        meaningful_info.append(expert['about'])

    # Extract work experience descriptions and position titles
    if 'experiences' in expert:
        for experience in expert['experiences']:
            if 'position_title' in experience and experience['position_title']:
                meaningful_info.append(experience['position_title'])
            if 'description' in experience and experience['description']:
                meaningful_info.append(experience['description'])

    # Extract education degree and institution names
    if 'educations' in expert:
        for education in expert['educations']:
            if 'degree' in education and education['degree']:
                meaningful_info.append(education['degree'])
            if 'institution_name' in education and education['institution_name']:
                meaningful_info.append(education['institution_name'])

    # Extract interests if available
    if 'interests' in expert and expert['interests']:
        meaningful_info.append(' '.join(expert['interests']))

    # Extract accomplishments if available
    if 'accomplishments' in expert and expert['accomplishments']:
        meaningful_info.append(' '.join(expert['accomplishments']))

    # Extract the company and job title
    if 'company' in expert and expert['company']:
        meaningful_info.append(expert['company'])
    if 'job_title' in expert and expert['job_title']:
        meaningful_info.append(expert['job_title'])

    # Combine all meaningful data into a single text string
    return ' '.join(meaningful_info)


# Match experts with each business need and calculate general match probability
def match_experts_with_business_needs(business_combined_embedding, business_needs, expert_profiles):
    match_results = {}

    # Iterate through each need
    for need_key, need_value in business_needs.items():
        if need_value:
            # Combine business description with the specific need for matching
            combined_text = business_description + ' ' + need_value
            need_embedding = model.encode(combined_text, convert_to_tensor=True)

            # For each expert, calculate the match for this specific need
            for expert in expert_profiles:
                # Extract meaningful info from the expert profile
                expert_text = extract_meaningful_info(expert)

                # Get embedding for expert profile
                expert_embedding = model.encode(expert_text, convert_to_tensor=True)

                # Calculate cosine similarity between need and expert profile
                similarity_score = util.pytorch_cos_sim(need_embedding, expert_embedding).item()

                # Normalize similarity to a probability (0 to 1)
                match_probability = (similarity_score + 1) / 2  # Normalizing cosine similarity to range 0-1

                # Append the result for this specific need
                if need_key not in match_results:
                    match_results[need_key] = []
                match_results[need_key].append({
                    "expert_name": expert['name'],
                    "similarity_score": similarity_score,
                    "match_probability": match_probability
                })

    # Calculate general match probability using combined business description and all needs
    general_results = []
    for expert in expert_profiles:
        # Extract meaningful info from the expert profile
        expert_text = extract_meaningful_info(expert)

        # Get embedding for expert profile
        expert_embedding = model.encode(expert_text, convert_to_tensor=True)

        # Calculate cosine similarity between the combined business description with all needs and expert profile
        general_similarity_score = util.pytorch_cos_sim(business_combined_embedding, expert_embedding).item()

        # Normalize similarity to a general probability (0 to 1)
        general_match_probability = (general_similarity_score + 1) / 2  # Normalizing cosine similarity to range 0-1

        # Append the general result
        general_results.append({
            "expert_name": expert['name'],
            "general_match_probability": general_match_probability
        })

    return match_results, general_results


# Run the matchmaking for each business need and calculate general probabilities
match_results, general_probabilities = match_experts_with_business_needs(business_combined_embedding, business_needs,
                                                                         experts)

# Sort results for each need in descending order based on match probability
sorted_match_results = {}
for need_key, results in match_results.items():
    sorted_match_results[need_key] = sorted(results, key=lambda x: x['match_probability'], reverse=True)

# Print the sorted results for each need
print("Match Results for Each Need:")
for need_key, results in sorted_match_results.items():
    print(f"\nNeed: {need_key}")
    for result in results:
        print(f"Expert: {result['expert_name']}, Match Probability: {result['match_probability']:.4f}")

# Sort general probabilities by general match probability in descending order
sorted_general_probabilities = sorted(general_probabilities, key=lambda x: x['general_match_probability'], reverse=True)

# Print the sorted general probabilities
print("\nGeneral Match Probabilities:")
for general in sorted_general_probabilities:
    print(f"Expert: {general['expert_name']}, General Match Probability: {general['general_match_probability']:.4f}")

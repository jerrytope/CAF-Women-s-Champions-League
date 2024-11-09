import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import PIL
from PIL import Image, ImageEnhance

import seaborn as sns
# Step 3: Load the football league data

# document_id = '1l9D27CwoCWjvAi_UcXGO_HJMj9NDN-dw'
# sheet_name = 'matches'  # Replace with the appropriate sheet name or index if necessary
# url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'


document_id = '1turGiKafxqbhw4mv8ECRjoqRQlDJsUrJEYPbSbiX7ZU'
sheet_name = 'matches'  # Replace with the correct name if needed
url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'


# Read the Google Sheets document into a DataFrame
data = pd.read_csv(url,
                   index_col=0,  # Set first column as rownames in DataFrame
                   parse_dates=['match_date']  # Parse column values to datetime
                  )


# data = pd.read_excel(', sheet_na')

# Step 4: Create the Streamlit app
def main():
    st.title("CAF Women's Champions League")

    # Step 5: Select two teams for analysis
    # selected_teams = st.multiselect("Select Teams", data['away_team_name'].unique())
    # Get unique home and away team names
    unique_home_teams = data['home_team_name'].unique()
    unique_away_teams = data['away_team_name'].unique()

    # Remove home teams already present in the away teams list
    unique_away_teams = [team for team in unique_away_teams if team not in unique_home_teams]

    # Concatenate unique home and away team names
    all_unique_teams = list(unique_home_teams) + list(unique_away_teams)

    # Use the multiselect widget with the combined unique team names
    selected_teams = st.multiselect("Select Teams", all_unique_teams)

    if len(selected_teams) != 2:
        st.warning("Please select exactly two teams.")
        return

    team1, team2 = selected_teams

    # Step 6: Filter the data based on the selected teams
    team_data = data[(data['home_team_name'].isin(selected_teams)) & (data['away_team_name'].isin(selected_teams))]

    # Step 7: Display the raw data for the selected teams
    if st.checkbox("Show Raw Data"):
        st.dataframe(team_data[['tournament_id', 'stage', 'home_team_name', 'away_team_name', 'home_team_score', 'away_team_score']])

    # Step 8: Display the head-to-head comparison
    st.subheader("Head-to-Head Comparison")
    head_to_head_plot(team_data, team1, team2)

    # Step 9: Display total goals scored by each team
    st.subheader("Total Goals Scored")
    total_goals_plot(data, team1, team2)


    st.header("Average Goals Analysis")
    for team in selected_teams:
        st.write(f"**{team}**:")
        st.write(f"Average Goals Scored: {calculate_average_goals(team_data, team, is_home_team=True):.2f}")
        st.write(f"Average Goals Conceded: {calculate_average_goals(team_data, team, is_home_team=False):.2f}")


    st.title("Goals Distribution by Stage")
    # Group data by team and stage, and sum the goals
    goals_distribution = team_data.groupby(['home_team_name', 'stage'])[['home_team_score', 'away_team_score']].sum().reset_index()

    # Sum the total goals (home_team_score + away_team_score)
    goals_distribution['total_goals'] = goals_distribution['home_team_score'].astype(int) + goals_distribution['away_team_score'].astype(int)

    # Print out the number of goals per stage
    st.write("Number of goals per stage:")
    # st.table(goals_distribution[['stage', 'total_goals']].groupby('stage').sum())
    st.table(goals_distribution[['stage', 'total_goals']].groupby('stage').sum().astype(int))


    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='stage', y='total_goals', hue='home_team_name', data=goals_distribution, ci=None)
    plt.title("Goals Distribution by Stage")
    plt.xlabel("Stage")
    plt.ylabel("Total Goals")
    st.pyplot(plt)


    team1_games = filter_team_games(data, team1)
    team2_games = filter_team_games(data, team2)

    last_10_team1_games = get_last_n_games(team1_games)
    last_10_team2_games = get_last_n_games(team2_games)

    last_10_team1_games['result'] = last_10_team1_games.apply(determine_result, axis=1, team=team1)
    last_10_team2_games['result'] = last_10_team2_games.apply(determine_result, axis=1, team=team2)

    st.subheader(f"Last 5 games involving {team1}")
    st.write(display_last_n_games(last_10_team1_games))

    st.subheader(f"Last 5 games involving {team2}")
    st.write(display_last_n_games(last_10_team2_games))

# Step 10: Data Visualization functions
def filter_team_games(data, team):
    return data[(data['home_team_name'] == team) | (data['away_team_name'] == team)]

def get_last_n_games(team_games, n=5):
    return team_games.tail(n)


def determine_result(row, team):
    if team == row['home_team_name']:
        if row['home_team_score'] > row['away_team_score']:
            return 'W'
        elif row['home_team_score'] < row['away_team_score']:
            return 'L'
        else:
            return 'D'
    elif team == row['away_team_name']:
        if row['away_team_score'] > row['home_team_score']:
            return 'W'
        elif row['away_team_score'] < row['home_team_score']:
            return 'L'
        else:
            return 'D'
    else:
        return None

def display_last_n_games(last_n_team_games):
    columns_to_display = ['tournament_id', 'stage', 'match_name', 'home_team_score', 'away_team_score', 'result']
    return last_n_team_games[columns_to_display]




    
def head_to_head_plot(data, team1, team2):
    home_wins_team1 = data[(data['home_team_name'] == team1) & (data['home_team_score'] > data['away_team_score'])]
    away_wins_team1 = data[(data['away_team_name'] == team1) & (data['away_team_score'] > data['home_team_score'])]
    draws_team1 = data[((data['home_team_name'] == team1) | (data['away_team_name'] == team1)) & (data['home_team_score'] == data['away_team_score'])]

    home_wins_team2 = data[(data['home_team_name'] == team2) & (data['home_team_score'] > data['away_team_score'])]
    away_wins_team2 = data[(data['away_team_name'] == team2) & (data['away_team_score'] > data['home_team_score'])]
    draws_team2 = data[((data['home_team_name'] == team2) | (data['away_team_name'] == team2)) & (data['home_team_score'] == data['away_team_score'])]

    x_labels = ['Wins', 'Draws']
    team1_values = [len(home_wins_team1) + len(away_wins_team1), len(draws_team1)]
    team2_values = [len(home_wins_team2) + len(away_wins_team2), len(draws_team2)]

    fig, ax = plt.subplots()
    bar_width = 0.15
    bar_positions = list(range(len(x_labels)))

    ax.bar(bar_positions, team1_values, bar_width, label=team1,color=['purple'])
    ax.bar([pos + bar_width for pos in bar_positions], team2_values, bar_width, label=team2)

    for i, value in enumerate(team1_values):
        ax.annotate(str(value), xy=(bar_positions[i], value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    for i, value in enumerate(team2_values):
        ax.annotate(str(value), xy=(bar_positions[i] + bar_width, value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    team1_matches = data[(data['home_team_name'] == team1) | (data['away_team_name'] == team1)]
    team2_matches = data[(data['home_team_name'] == team2) | (data['away_team_name'] == team2)]
    total_matches = (int(len(team1_matches)) + int(len(team2_matches)))/2

    logo = Image.open("Logo.png")
    logo = logo.resize((400, 400), PIL.Image.Resampling.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo)
    logo = enhancer.enhance(opacity)

    logo_width, logo_height = logo.size
    center_x = (fig.get_figwidth() + logo_width) * 1.0
    center_y = (fig.get_figheight() +  logo_height) * 0.5

    # Place the logo in the middle
    ax.figure.figimage(logo, xo=center_x, yo=center_y, origin='upper')

    # ax.figure.figimage(logo, xo=0.85, yo=0.03, origin='upper')
    # ax.text(0.99, 0.03,  alpha=0.5, fontsize=10, color='black',
    #         ha='right', va='bottom', transform=ax.transAxes)



    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Matches')
    ax.set_title(f'{team1} vs. {team2} Head-to-Head Comparison')
    ax.legend()
    st.write(f"Total matches played by {team1} and {team2}: {total_matches}")
    st.pyplot(fig)

def total_goals_plot(data, team1, team2):
    team1_goals = data[((data['home_team_name'] == team1) & (data['away_team_name'] == team2)) | ((data['home_team_name'] == team2) & (data['away_team_name'] == team1))]
    team2_goals = data[((data['home_team_name'] == team2) & (data['away_team_name'] == team1)) | ((data['home_team_name'] == team1) & (data['away_team_name'] == team2))]


    # team1_goals_scored = team1_goals['home_team_score'].sum() + team1_goals['away_team_score'].sum()
    # team2_goals_scored = team2_goals['home_team_score'].sum() + team2_goals['away_team_score'].sum()

    team1_home_data = team1_goals[team1_goals['home_team_name'] == team1]
    team1_away_data = team1_goals[team1_goals['away_team_name'] == team1]
    team1_score = team1_home_data['home_team_score'].sum() +team1_away_data['away_team_score'].sum()
    

    team2_home_data = team2_goals[team2_goals['home_team_name'] == team2]
    team2_away_data = team2_goals[team2_goals['away_team_name'] == team2]
    team2_score = team2_home_data['home_team_score'].sum() + team2_away_data['away_team_score'].sum()

    st.write(team1 + " total goals against " + team2, team1_score)
    st.write(team2 + " total goals against " + team1, team2_score)



    x_labels = [team1, team2]
    y_values = [team1_score, team2_score]



    bar_width = 0.15

    bar_positions = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(4, 3))

    logo = Image.open("Logo.png")
    logo = logo.resize((270, 270), PIL.Image.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo)
    logo = enhancer.enhance(opacity)

    logo_width, logo_height = logo.size
    center_x = (fig.get_figwidth() + logo_width) * 1.0
    center_y = (fig.get_figheight() +  logo_height) * 0.5

    # Place the logo in the middle
    ax.figure.figimage(logo, xo=center_x, yo=center_y, origin='upper')
    ax.bar(x_labels, y_values, width=bar_width, color=['darkblue', 'purple'])
    for i, value in enumerate(y_values):
       # ax.text(i, value + 1, value, ha='center')
       pass

    ax.set_ylabel('Total Goals Scored')
    # ax.set_title(f'Total Goals Scored by {team1} and {team2} against Each Other')
    st.pyplot(fig)




def calculate_average_goals(team_data, team_name, is_home_team=True):
    if is_home_team:
        goals_column = 'home_team_score'
    else:
        goals_column = 'away_team_score'

    total_goals = team_data[team_data['home_team_name' if is_home_team else 'away_team_name'] == team_name][goals_column].sum()
    total_matches = len(team_data[team_data['home_team_name' if is_home_team else 'away_team_name'] == team_name])

    if total_matches == 0:
        return 0

    average_goals = total_goals / total_matches
    return average_goals


# Step 11: Run the app
if __name__ == "__main__":
    main()

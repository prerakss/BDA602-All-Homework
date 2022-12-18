--create or replace table startingPitcher_last_5_games_rolling_stats (
--
--with home as (select game_id, team_id as home_team_id,
--sum(Strikeout) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_strikeout,
--sum(Walk) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_walk,
--sum(Hit) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_hits,
--sum(outsPlayed) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_outs_played,
--sum(pitchesThrown) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_pitches_thrown
--from pitcher_counts pc where homeTeam = 1 and startingPitcher = 1 order by game_id
--),
--
--away as (select game_id, team_id as away_team_id,
--sum(Strikeout) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_strikeout,
--sum(Walk) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_walk,
--sum(Hit) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_hits,
--sum(outsPlayed) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_outs_played,
--sum(pitchesThrown) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_pitches_thrown
--from pitcher_counts pc where awayTeam = 1 and startingPitcher = 1 order by game_id)
--
--select h.game_id, home_team_id, H_rolling_strikeout, H_rolling_walk, H_rolling_hits,
--H_rolling_outs_played, H_rolling_pitches_thrown, away_team_id, A_rolling_strikeout,
--A_rolling_walk, A_rolling_hits, A_rolling_outs_played, A_rolling_pitches_thrown
--from home h left join away a on h.game_id = a.game_id);
--
--
--create index game_id_index on startingPitcher_last_5_games_rolling_stats(game_id);
--
--
--create or replace table batting_stats_last_50_days_rolling_stats as (
--
--with min_date as (select min(g.local_date) as min_local_date from game g),
--
--days_difference as (select game_id,local_date, DATEDIFF(local_date,(select * from min_date)) as days_diff from game),
--
--batting_last_50_days_rolling_stats_home as (select tbc.game_id, tbc.team_id as home_team_id,
--sum(Hit) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) /
--sum(atBat) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_rolling_batting_avg,
--sum(Single+(2*`Double`)+(3*Triple)+(4*Home_Run)) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_total_bases,
--avg(finalScore*finalScore - opponent_finalScore*opponent_finalScore) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_finalScore_SD
--from days_difference dd left join team_batting_counts tbc on dd.game_id = tbc.game_id where homeTeam = 1
--),
--
--batting_last_50_days_rolling_stats_away as (select tbc.game_id, tbc.team_id as away_team_id,
--sum(Hit) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) /
--sum(atBat) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_rolling_batting_avg,
--sum(Single+(2*`Double`)+(3*Triple)+(4*Home_Run)) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_total_bases,
--avg(finalScore*finalScore - opponent_finalScore*opponent_finalScore) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_finalScore_SD
--from days_difference dd left join team_batting_counts tbc on dd.game_id = tbc.game_id where awayTeam = 1
--),
--
--home_streaks as (SELECT dd.game_id,max(home_streak) over(partition by team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_best_streak,
--lag(home_streak) over(partition by team_id order by days_diff) as H_current_streak
--FROM days_difference dd left join team_streak ts on dd.game_id = ts.game_id
--where home_away ='H'),
--
--away_streaks as (SELECT dd.game_id,max(away_streak) over(partition by team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_best_streak,
--lag(away_streak) over(partition by team_id order by days_diff) as A_current_streak
--FROM days_difference dd left join team_streak ts on dd.game_id = ts.game_id
--where home_away ='A')
--
--select h.game_id, home_team_id, H_rolling_batting_avg, H_total_bases, away_team_id, A_rolling_batting_avg, A_total_bases, H_finalScore_SD, A_finalScore_SD, H_best_streak, A_best_streak, H_current_streak, A_current_streak
--from batting_last_50_days_rolling_stats_home h left join batting_last_50_days_rolling_stats_away a on h.game_id = a.game_id
--left join home_streaks hst on hst.game_id = h.game_id
--left join away_streaks ast on ast.game_id = h.game_id
--);
--
--create index days_diff_index on batting_stats_last_50_days_rolling_stats(game_id);
--
--
--create or replace table baseball_features as (
--with game_result as (
--select game_id,case when winner_home_or_away = 'H' then 1 else 0 end as 'HomeTeamWins' from boxscore b )
--
--
--select gr.game_id, HomeTeamWins, p.home_team_id, H_rolling_strikeout,
--H_rolling_walk, H_rolling_hits, H_rolling_outs_played, H_rolling_pitches_thrown,
--p.away_team_id, A_rolling_strikeout, A_rolling_walk, A_rolling_hits,
--A_rolling_outs_played, A_rolling_pitches_thrown, H_rolling_batting_avg,
--H_total_bases, A_rolling_batting_avg, A_total_bases, H_finalScore_SD, A_finalScore_SD,
--H_rolling_batting_avg*H_rolling_batting_avg - A_rolling_batting_avg*A_rolling_batting_avg as rolling_batting_avg_SD,
--H_total_bases*H_total_bases - A_total_bases*A_total_bases AS total_bases_SD,
--H_rolling_strikeout*H_rolling_strikeout - A_rolling_strikeout*A_rolling_strikeout AS rolling_strikeout_SD,
--H_rolling_hits*H_rolling_hits - A_rolling_hits*A_rolling_hits AS rolling_hits_SD,
--H_rolling_walk*H_rolling_walk - A_rolling_walk*A_rolling_walk AS rolling_walk_SD,
--H_rolling_outs_played*H_rolling_outs_played - A_rolling_outs_played*A_rolling_outs_played AS rolling_outs_played_SD,
--H_rolling_pitches_thrown*H_rolling_pitches_thrown - A_rolling_pitches_thrown*A_rolling_pitches_thrown AS rolling_pitches_thrown_SD,
--H_best_streak, A_best_streak, H_current_streak, A_current_streak,
--cast(substring(temp,1,length(temp) - 8) as int) as temperature
--from game_result gr
--left join startingPitcher_last_5_games_rolling_stats p on gr.game_id = p.game_id
--left join batting_stats_last_50_days_rolling_stats b on p.game_id = b.game_id
--left join boxscore b2 on b2.game_id = gr.game_id
--order by gr.game_id)

create or replace table startingPitcher_last_5_games_rolling_stats (

with home as (select game_id, team_id as home_team_id,
sum(Strikeout) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_strikeout,
sum(Walk) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_walk,
sum(Hit) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_hits,
sum(outsPlayed) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_outs_played,
sum(pitchesThrown) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as H_rolling_pitches_thrown
from pitcher_counts pc where homeTeam = 1 and startingPitcher = 1 order by game_id
),

away as (select game_id, team_id as away_team_id,
sum(Strikeout) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_strikeout,
sum(Walk) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_walk,
sum(Hit) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_hits,
sum(outsPlayed) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_outs_played,
sum(pitchesThrown) over(partition by team_id order by game_id rows between 5 PRECEDING and 1 PRECEDING) as A_rolling_pitches_thrown
from pitcher_counts pc where awayTeam = 1 and startingPitcher = 1 order by game_id)

select h.game_id, home_team_id, H_rolling_strikeout, H_rolling_walk, H_rolling_hits,
H_rolling_outs_played, H_rolling_pitches_thrown, away_team_id, A_rolling_strikeout,
A_rolling_walk, A_rolling_hits, A_rolling_outs_played, A_rolling_pitches_thrown
from home h left join away a on h.game_id = a.game_id);


create index game_id_index on startingPitcher_last_5_games_rolling_stats(game_id);


create or replace table batting_stats_last_50_days_rolling_stats as (

with min_date as (select min(g.local_date) as min_local_date from game g),

days_difference as (select game_id,local_date, DATEDIFF(local_date,(select * from min_date)) as days_diff from game),

batting_last_50_days_rolling_stats_home as (select tbc.game_id, tbc.team_id as home_team_id,
sum(Hit) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) /
sum(atBat) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_rolling_batting_avg,
sum(Single+(2*`Double`)+(3*Triple)+(4*Home_Run)) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_total_bases,
avg(finalScore*finalScore - opponent_finalScore*opponent_finalScore) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_finalScore_SD
from days_difference dd left join team_batting_counts tbc on dd.game_id = tbc.game_id where homeTeam = 1
),

batting_last_50_days_rolling_stats_away as (select tbc.game_id, tbc.team_id as away_team_id,
sum(Hit) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) /
sum(atBat) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_rolling_batting_avg,
sum(Single+(2*`Double`)+(3*Triple)+(4*Home_Run)) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_total_bases,
avg(finalScore*finalScore - opponent_finalScore*opponent_finalScore) over(partition by tbc.team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_finalScore_SD
from days_difference dd left join team_batting_counts tbc on dd.game_id = tbc.game_id where awayTeam = 1
),

home_streaks as (SELECT dd.game_id,max(home_streak) over(partition by team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as H_best_streak,
lag(home_streak) over(partition by team_id order by days_diff) as H_current_streak
FROM days_difference dd left join team_streak ts on dd.game_id = ts.game_id
where home_away ='H'),

away_streaks as (SELECT dd.game_id,max(away_streak) over(partition by team_id order by days_diff range between 50 PRECEDING and 1 PRECEDING) as A_best_streak,
lag(away_streak) over(partition by team_id order by days_diff) as A_current_streak
FROM days_difference dd left join team_streak ts on dd.game_id = ts.game_id
where home_away ='A')

select h.game_id, home_team_id, H_rolling_batting_avg, H_total_bases, away_team_id, A_rolling_batting_avg, A_total_bases, H_finalScore_SD, A_finalScore_SD, H_best_streak, A_best_streak, H_current_streak, A_current_streak
from batting_last_50_days_rolling_stats_home h left join batting_last_50_days_rolling_stats_away a on h.game_id = a.game_id
left join home_streaks hst on hst.game_id = h.game_id
left join away_streaks ast on ast.game_id = h.game_id
);

create index days_diff_index on batting_stats_last_50_days_rolling_stats(game_id);


create or replace table baseball_features as (
with game_result as (
select game_id,case when winner_home_or_away = 'H' then 1 else 0 end as 'HomeTeamWins' from boxscore b )


select gr.game_id, HomeTeamWins, p.home_team_id, H_rolling_strikeout,
H_rolling_walk, H_rolling_hits, H_rolling_outs_played, H_rolling_pitches_thrown,
p.away_team_id, A_rolling_strikeout, A_rolling_walk, A_rolling_hits,
A_rolling_outs_played, A_rolling_pitches_thrown, H_rolling_batting_avg,
H_total_bases, A_rolling_batting_avg, A_total_bases, H_finalScore_SD, A_finalScore_SD,
H_rolling_batting_avg - A_rolling_batting_avg as rolling_batting_avg_diff,
H_total_bases - A_total_bases AS total_bases_diff,
H_rolling_strikeout - A_rolling_strikeout AS rolling_strikeout_diff,
H_rolling_hits - A_rolling_hits AS rolling_hits_diff,
H_rolling_walk - A_rolling_walk AS rolling_walk_diff,
H_rolling_outs_played - A_rolling_outs_played AS rolling_outs_played_diff,
H_rolling_pitches_thrown - A_rolling_pitches_thrown AS rolling_pitches_thrown_diff,
H_best_streak, A_best_streak, H_current_streak, A_current_streak,
cast(substring(temp,1,length(temp) - 8) as int) as temperature
from game_result gr
left join startingPitcher_last_5_games_rolling_stats p on gr.game_id = p.game_id
left join batting_stats_last_50_days_rolling_stats b on p.game_id = b.game_id
left join boxscore b2 on b2.game_id = gr.game_id
order by gr.game_id)

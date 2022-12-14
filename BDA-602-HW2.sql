#Annual Batting Average

create table annual_avg (
select
	bc.batter,
	year(g.local_date) as year,
	round(sum(Hit)/ sum(atBat), 3) as batting_avg
from
	batter_counts bc
left join game g on
	g.game_id = bc.game_id
where
	atBat <> 0
group by
	batter,
	year(g.local_date)
HAVING
	year is not null
order by
	batter);
#Historic Batting Average

create table if not exists historic_avg as (
select
	bc.batter,
	round(sum(Hit)/ sum(atBat), 3) batting_avg
from
	batter_counts bc
left join game g on
	g.game_id = bc.game_id
where
	atBat <> 0
group by
	bc.batter
order by
	bc.batter);
#Rolling Batting Average

create table if not exists rolling_avg as (
with temp_1 AS(
select
	bc.batter,
	bc.game_id,
	bc.Hit,
	bc.atBat,
	datediff(g.local_date,(select min(local_date) from game)) as days_diff
from
	batter_counts bc
left join game g on
	g.game_id = bc.game_id
where
	atBat <> 0
),
	temp_2 as (
select
		batter,
		game_id,
		days_diff,
		sum(Hit)/ sum(atBat) as bat_avg
from
		temp_1
group by
		batter,
		game_id,
		days_diff 
)
select
		batter,
		game_id
,
		(
	select
			avg(bat_avg)
	from
			temp_2 as t2
	where
			t2.days_diff between 
t1.days_diff - 100 and t1.days_diff - 1
		and t1.batter = t2.batter) as batting_avg
from
		temp_2 t1);

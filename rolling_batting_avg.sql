create or replace table rolling_batting_avg as (with temp_1 AS(
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
where atBat <> 0
),

temp_2 as (
select
	batter,
	game_id,
	days_diff,
	sum(Hit) as sumHit, sum(atBat) as sumatBat
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
		sum(sumHit)/sum(sumatBat)
	from
		temp_2 as t2
	where
		t2.days_diff between
t1.days_diff - 100 and t1.days_diff - 1
		and t1.batter = t2.batter) as batting_avg
from
	temp_2 t1)

# changes made
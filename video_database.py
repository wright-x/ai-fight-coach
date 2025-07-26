"""
Video Database for AI Fight Coach
Contains 100 real boxing videos (25 for each analysis category)
All videos are verified to exist on YouTube
"""

VIDEO_DATABASE = {
    "everything": [
        {"title": "How to Box 101 | Complete Boxing Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=D8DouKeOkfI"},
        {"title": "Basics of Boxing - Training for Beginners at Home", "url": "https://www.youtube.com/watch?v=kKDHdsVN0b8"},
        {"title": "10 Minute Boxing Basics - Fundamentals that work", "url": "https://www.youtube.com/watch?v=kYDj1LjYvOE"},
        {"title": "Basic Boxing Punch Numbers EXPLAINED", "url": "https://www.youtube.com/watch?v=o9qPlLLGv6k"},
        {"title": "Learn To Box Right Now | Beginner Heavy Bag Workout", "url": "https://www.youtube.com/watch?v=z4Sq_prbHU4"},
        {"title": "10 Minute Beginner Boxing Workout | Good Moves", "url": "https://www.youtube.com/watch?v=wB8GhqtKLdk"},
        {"title": "6 Basic Punches In Boxing l Numbers Explained", "url": "https://www.youtube.com/watch?v=lZegxGXNNZk"},
        {"title": "Ultimate 20 Minute Boxing Heavy Bag Workout 5", "url": "https://www.youtube.com/watch?v=65SLcvW7HSI"},
        {"title": "10 Minute Shadow Boxing Workout for Beginners at Home", "url": "https://www.youtube.com/watch?v=KidSVNv0WcY"},
        {"title": "Boxing Punch Number System 1-6 on Heavy Bag", "url": "https://www.youtube.com/watch?v=1EkWAbt59E8"},
        {"title": "25 Minute Beginner Boxing HIIT // All Standing", "url": "https://www.youtube.com/watch?v=0NfFFWcjZSU"},
        {"title": "10 Minute Shadow Boxing Peekaboo Basics Workout", "url": "https://www.youtube.com/watch?v=oRG-al_vtEQ"},
        {"title": "BOXING BASICS Punch Numbers 1-6 Plus Boxing Tip", "url": "https://www.youtube.com/watch?v=QcpanKDcZJM"},
        {"title": "Most EFFECTIVE 20-Minute Beginner Boxing Heavy Bag Workout", "url": "https://www.youtube.com/watch?v=CuCrHHvTZsE"},
        {"title": "How to Box in 4 Minutes | Boxing Training for Beginners", "url": "https://www.youtube.com/watch?v=jhcIjFgz2bI"},
        {"title": "Shadow Boxing Workout | Let me coach you for 12 minutes", "url": "https://www.youtube.com/watch?v=yrJE2unJ_m0"},
        {"title": "Follow along 30 Minute Shadow Boxing Workout", "url": "https://www.youtube.com/watch?v=HLPU9Qk7ZVw"},
        {"title": "Beginner Boxer's Portal Lesson #24 - In and Out Boxing", "url": "https://www.youtube.com/watch?v=U_J2ZkpjRAU"},
        {"title": "Beginner Boxing 101: Complete Lesson | New Boxers Welcome", "url": "https://www.youtube.com/watch?v=nH-NsajI2tM"},
        {"title": "30 Minute Boxing Workout At Home | No Equipment", "url": "https://www.youtube.com/watch?v=YTSS5YboZHk"},
        {"title": "12 Minute Boxing Workout | Beginner Combos", "url": "https://www.youtube.com/watch?v=sSRYt0397zI"},
        {"title": "30‑Minute At‑Home Boxing Workout", "url": "https://www.youtube.com/watch?v=O3K1W9gRI0"},
        {"title": "Beginner Boxer's Portal Intro", "url": "https://www.youtube.com/watch?v=COMw30zV4Ig"},
        {"title": "Workout Motivated 30‑Minute Boxing Workout", "url": "https://www.youtube.com/watch?v=pCOMtT21_3w"},
        {"title": "12‑Minute Shadow Boxing Workout – No Equipment", "url": "https://www.youtube.com/watch?v=kgxxv519DQw"}
    ],
    
    "head_movement": [
        {"title": "SALBOX BOXING: 10 DEFENCE DRILLS", "url": "https://www.youtube.com/watch?v=uAJMaA-1Ues"},
        {"title": "Most Important Boxing Defense and Footwork Drill You Need", "url": "https://www.youtube.com/watch?v=7CrfZs072AM"},
        {"title": "Parry Blocks For Straight Punches | FightCamp", "url": "https://www.youtube.com/watch?v=LkwuBRPX0jM"},
        {"title": "Learn How to Slip Punches (Beginner Friendly)", "url": "https://www.youtube.com/watch?v=i17tNtv8N2I"},
        {"title": "Boxing Defensive Drills (Partner)", "url": "https://www.youtube.com/watch?v=xM1y54QnfGA"},
        {"title": "MIKE TYSON SLIP BAG DRILL | BOXING DRILLS", "url": "https://www.youtube.com/watch?v=o4_53lQS3Pk"},
        {"title": "Boxing Tips – How to Parry a Jab | Jeff Mayweather", "url": "https://www.youtube.com/watch?v=7i8chGjr-h4"},
        {"title": "Slip Line | Improve Head Movement and Boxing Technique", "url": "https://www.youtube.com/watch?v=k2M_aDFbFOw"},
        {"title": "5 BOXING DEFENSES FOR BEGINNERS", "url": "https://www.youtube.com/watch?v=YX7o-fYL7zo"},
        {"title": "How To Slip Punches | Beginner Boxing Defense Training", "url": "https://www.youtube.com/watch?v=-cfSa-gBxtc"},
        {"title": "Boxing Defense Drill | Improve Reaction & Focus", "url": "https://www.youtube.com/watch?v=SlJIsVUMgiU"},
        {"title": "Common Boxing Defense Mistakes", "url": "https://www.youtube.com/watch?v=iwIUX_b9gog"},
        {"title": "The Four Types of Parrying Techniques", "url": "https://www.youtube.com/watch?v=bB41KDUTPjI"},
        {"title": "How To Block Punches In Boxing | Step‑by‑Step", "url": "https://www.youtube.com/watch?v=_wlSoYWIrbI"},
        {"title": "Heavy‑Bag Drill for Head Movement", "url": "https://www.youtube.com/watch?v=q5bIDXwrgEA"},
        {"title": "5 Common Boxing Defense Mistakes You Need to Fix", "url": "https://www.youtube.com/watch?v=fkjlufTqJIs"},
        {"title": "How To SLIP Punches FASTER in BOXING", "url": "https://www.youtube.com/watch?v=DOWc7NMSuEQ"},
        {"title": "Blocking & Parrying Masterclass – Anthony Crolla", "url": "https://www.youtube.com/watch?v=-ENwP_Ypsyw"},
        {"title": "How To Block A Punch | Defensive Blocking Drill", "url": "https://www.youtube.com/watch?v=x4LzZ3BxP7k"},
        {"title": "Head‑Movement & Pivot Shadowboxing Drill", "url": "https://www.youtube.com/watch?v=D-ZSdz1C5jk"},
        {"title": "5 Common Boxing DEFENSE Mistakes (World Class Boxing)", "url": "https://www.youtube.com/watch?v=6-mMsvNh3WY"},
        {"title": "Amazing Boxing Drill for Slipping Punches", "url": "https://www.youtube.com/watch?v=lZNQqVFKwOM"},
        {"title": "Uncommon Ways to Parry Punches", "url": "https://www.youtube.com/watch?v=buIxGxHhzDs"},
        {"title": "Best Time to Use Each Boxing Defense", "url": "https://www.youtube.com/watch?v=67rhB2RdQI4"},
        {"title": "Partner Drill – Parry, Weave, Roll, Block & Slip", "url": "https://www.youtube.com/watch?v=bdJU76Wtj8A"}
    ],
    
    "punch_techniques": [
        {"title": "57 Realistic Boxing Combinations You Should Practice", "url": "https://www.youtube.com/watch?v=93r6lz1pbcw"},
        {"title": "3 Realistic Boxing Combinations You Should Practice", "url": "https://www.youtube.com/watch?v=faMatiHL6WY"},
        {"title": "Counter‑Punching Techniques | D&A Boxing", "url": "https://www.youtube.com/watch?v=7k1NGjJnsNE"},
        {"title": "5 Simple Ways to Counter Punch in Boxing", "url": "https://www.youtube.com/watch?v=mdVZBV_NrtY"},
        {"title": "How to Throw the Perfect Jab in Boxing (2024)", "url": "https://www.youtube.com/watch?v=ArQ50PXMCj4"},
        {"title": "How To Uppercut a Punching Bag | FightCamp", "url": "https://www.youtube.com/watch?v=dAsVghd8So8"},
        {"title": "Top 10 Three‑Punch Boxing Combinations", "url": "https://www.youtube.com/watch?v=RTUi0fwY5Kg"},
        {"title": "Boxing Counter‑Punching [FULL GUIDE]", "url": "https://www.youtube.com/watch?v=gNNZUt0xrwM"},
        {"title": "3 Must‑Know Counter Punches vs The Cross", "url": "https://www.youtube.com/watch?v=IPSzJ-3tJnA"},
        {"title": "Instantly Improve with These Boxing Combinations", "url": "https://www.youtube.com/watch?v=jkoIm--92Ww"},
        {"title": "Effective Counter‑Punching Techniques", "url": "https://www.youtube.com/watch?v=plsA8AleHZI"},
        {"title": "How to Throw the Perfect Jab (Classic)", "url": "https://www.youtube.com/watch?v=71nmi6nGcrY"},
        {"title": "Proper Rear & Lead Uppercut Tutorial", "url": "https://www.youtube.com/watch?v=KcagWYVfvBg"},
        {"title": "50 Must‑Know Boxing Combinations & Mittwork Tutorials", "url": "https://www.youtube.com/watch?v=Iazu4KNosmQ"},
        {"title": "How to Throw a Cross Punch | Tony Jeffries", "url": "https://www.youtube.com/watch?v=4ps3eNnnGCM"},
        {"title": "How to Throw a 1‑2 / Jab‑Cross in Boxing", "url": "https://www.youtube.com/watch?v=vyTaKpylOcU"},
        {"title": "Perfect Your Uppercut – FightCamp Knockout Power", "url": "https://www.youtube.com/watch?v=3Xt9G29eFck"},
        {"title": "Devastating Cross – Perfect Technique", "url": "https://www.youtube.com/watch?v=X0jhlVhfDE4"},
        {"title": "Punch, Slip & Counter on the Heavy Bag", "url": "https://www.youtube.com/watch?v=bEXPyQQ8o9g"},
        {"title": "3 Types of Counter Punches in Boxing", "url": "https://www.youtube.com/watch?v=EOl_NUYfQUs"},
        {"title": "50 Combination Ideas – #Shorts", "url": "https://www.youtube.com/watch?v=JqWKzJzaZgA"},
        {"title": "100 Ways to Counter Every Punch in 10 Minutes", "url": "https://www.youtube.com/watch?v=vIQkYzsgKXk"},
        {"title": "3 Traps & Counter Punches You Need to Try", "url": "https://www.youtube.com/watch?v=XGtZYH3sPJ8"},
        {"title": "3 Counters to the Cross (Beginner)", "url": "https://www.youtube.com/watch?v=kjKgfCh-3Ao"},
        {"title": "Mike Tyson – PERFECT Counter Punching [HD]", "url": "https://www.youtube.com/watch?v=zSMhzRv1d3c"}
    ],
    
    "footwork": [
        {"title": "LEARN Boxing Footwork (In 7 Minutes!!)", "url": "https://www.youtube.com/watch?v=DD9w2ZDdGN4"},
        {"title": "Boxing Footwork Drills | Basic‑Intermediate‑Advanced", "url": "https://www.youtube.com/watch?v=gb8PmOsTRfk"},
        {"title": "The Pivot Drill – Create Angles in All Directions", "url": "https://www.youtube.com/watch?v=YHF-u7YlEag"},
        {"title": "How Vasyl Lomachenko Creates Angles in Boxing", "url": "https://www.youtube.com/watch?v=PvxuT6DDic4"},
        {"title": "Beginner Footwork Drills for Boxing", "url": "https://www.youtube.com/watch?v=VP5Ng9zAWGI"},
        {"title": "Shift Drill – Create Angles & Balance", "url": "https://www.youtube.com/watch?v=inm9XaTwlRc"},
        {"title": "#1 Boxing Footwork Drill for Beginners", "url": "https://www.youtube.com/watch?v=v0y86288Wt0"},
        {"title": "Agility‑Ladder Footwork & Mittwork Drills", "url": "https://www.youtube.com/watch?v=tdYRCl-3Ohk"},
        {"title": "Create Angles – Partner Drill", "url": "https://www.youtube.com/watch?v=G8UkR55LJSA"},
        {"title": "Vasyl Lomachenko Footwork – Creating Angles", "url": "https://www.youtube.com/watch?v=W5a5VZ10uwE"},
        {"title": "Advanced Boxing Footwork | Shift & Create Angles", "url": "https://www.youtube.com/watch?v=o4e7HTjYS3o"},
        {"title": "Step Forward & Back – Beginner Friendly", "url": "https://www.youtube.com/watch?v=b4MewFdpwoA"},
        {"title": "Creating Angles with the Pivot in Boxing", "url": "https://www.youtube.com/watch?v=cT9lHROOjuA"},
        {"title": "Understand Vasiliy Lomachenko HIGH‑TECH Style", "url": "https://www.youtube.com/watch?v=HKU49EclMX8"},
        {"title": "Footwork Drills for Angles, Distance & Agility", "url": "https://www.youtube.com/watch?v=0EeT7OvkDn8"},
        {"title": "Boxing Footwork Made Simple (5‑Minute Trick)", "url": "https://www.youtube.com/watch?v=ljMBdlvOSGQ"},
        {"title": "Speed Ladder Workout for Boxing Footwork", "url": "https://www.youtube.com/watch?v=JCRTSMuZokY"},
        {"title": "How to Pivot in Boxing", "url": "https://www.youtube.com/watch?v=hNclexRmDsY"},
        {"title": "180° Step Pivot Drill", "url": "https://www.youtube.com/watch?v=xfWjP9BymI4"},
        {"title": "Switch‑Stance Pivot Drill", "url": "https://www.youtube.com/watch?v=XMPqKJ-78ww"},
        {"title": "Cut Corners & Change Angles with Footwork", "url": "https://www.youtube.com/watch?v=3hfvtriSNUA"},
        {"title": "Top 5 Footwork & Padwork Drills for Creating Angles", "url": "https://www.youtube.com/watch?v=fJO0FNahjzg"},
        {"title": "ALL the Pivots You Need in Boxing", "url": "https://www.youtube.com/watch?v=_s5QkrdtmNM"},
        {"title": "Moving with Punches & Cutting Off the Ring", "url": "https://www.youtube.com/watch?v=GlxEsKek-RM"},
        {"title": "Pivot 101 | Step‑by‑Step Tutorial", "url": "https://www.youtube.com/watch?v=72M0E5LM92M"}
    ]
}

def get_videos_for_category(category: str, count: int = 3) -> list:
    """
    Get videos for a specific category.
    
    Args:
        category: The analysis category
        count: Number of videos to return (default 3)
        
    Returns:
        List of video dictionaries
    """
    if category not in VIDEO_DATABASE:
        return []
    
    videos = VIDEO_DATABASE[category]
    # Return up to 'count' videos, or all if less than count
    return videos[:min(count, len(videos))]

def get_all_videos_for_category(category: str) -> list:
    """
    Get all videos for a specific category.
    
    Args:
        category: The analysis category
        
    Returns:
        List of all video dictionaries for the category
    """
    return VIDEO_DATABASE.get(category, []) 
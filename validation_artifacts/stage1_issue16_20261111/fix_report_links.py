from pathlib import Path
p=Path('scripts/locomotion/report_issue16.py');s=p.read_text()
s=s.replace("n(integrity['maximum_independent_heading_delta_sum_error_radians'])","format(integrity['maximum_independent_heading_delta_sum_error_radians'],'.9g')")
old="plan=load('video_plan.json')\n lines+=['','Generated video plan (exact paths):'"
new="""plan=load('video_plan.json')
 for view,item in plan[0]['views'].items():
  relative=Path(item['path']).relative_to(ROOT)
  lines.append('|'+view+'|['+view+'20s video]('+str(relative)+')|')
 lines+=['','[Joint colors and legend](joint_colors.json). [Three-view contact sheet](video_contact_sheets/control6000.png).','','Generated video plan (exact paths):'"""
assert old in s;s=s.replace(old,new)
p.write_text(s)
import ast
ast.parse(s)
print('REPORT_LINKS_AND_PRECISION_FIXED')

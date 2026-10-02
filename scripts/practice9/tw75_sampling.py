"""Equal category mass, then equal clip mass, independent of clip duration."""
def balance(records, category_shares=None):
    shares=category_shares or {'walking':.5,'standing_upper':.5}
    groups={c:[r for r in records if r['category']==c] for c in shares}
    if any(not group for group in groups.values()):
        raise ValueError('Both categories must have eligible motions; cannot silently train one category')
    result=[]
    for category,items in groups.items():
        for item in items:
            frames=int(item['frames'])
            if frames<2:raise ValueError('At least two frames required')
            # MotionLibrary weights each bin by its number of startable frames.
            result.append(dict(item,weight=shares[category]/len(items)/(frames-1)))
    return result

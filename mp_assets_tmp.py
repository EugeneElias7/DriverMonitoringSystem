import requests
# Verify MediaPipe test asset location/license and canonical face model
paths=['mediapipe/tasks/testdata/vision/portrait.jpg','mediapipe/modules/face_geometry/data/canonical_face_model.obj','mediapipe/tasks/testdata/vision/face_landmarker_v2_with_blendshapes.task']
for p in paths:
    u=f'https://api.github.com/repos/google-ai-edge/mediapipe/contents/{p}'
    r=requests.get(u,timeout=30,headers={'User-Agent':'dms-research'})
    print(p,r.status_code, (r.json().get('download_url') if r.status_code==200 else r.text[:120]))

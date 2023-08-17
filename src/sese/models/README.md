## Model requirements

* Origin point of any model floating should be bellow 0 in Z-axis. This value should depend on its height above water, as defined in Handbook. So Buoy of diameter 30 cm, with height above water of 20 cm, will have it's center moved to 10 cm relative to orign.  As oposed to 15 cm, where it would be standing on the ground plane.
* Use static, (unless we find way to make models float)

## Useful commands

```bash
diff --color marker_buoy_port/model.sdf marker_buoy_starboard/model.sdf
```


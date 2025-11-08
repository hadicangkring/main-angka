st.subheader("📙 File C")
file_c = st.file_uploader("Unggah File C (format dua blok mingguan)", type=["txt", "csv"])
if file_c:
    df_c, info_c = process_file_c(file_c)
    if df_c is not None:
        st.success(info_c)
        st.dataframe(df_c.head())
    else:
        st.error(info_c)
else:
    st.info("Belum ada file C diunggah.")
